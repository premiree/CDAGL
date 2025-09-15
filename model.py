import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torchdiffeq import odeint
import numpy as np
from transformers import AutoModel, AutoTokenizer

class FeatureExtractor(nn.Module):
    """Extract features from raw modalities"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Audio: AudioCLIP encoder (frozen, only for feature extraction)
        # In practice, you would load the actual AudioCLIP model
        self.audio_encoder = nn.Sequential(
            nn.Linear(config.audio_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, config.audio_hidden_dim)
        )
        
        # Visual: DenseNet-based encoder
        self.visual_encoder = nn.Sequential(
            nn.Linear(config.visual_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, config.visual_hidden_dim)
        )
        
        # Text: BERT encoder (frozen)
        self.text_encoder = AutoModel.from_pretrained('bert-base-uncased')
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.text_projection = nn.Linear(768, config.text_hidden_dim)
        
        # Speaker embedding
        self.speaker_embedding = nn.Embedding(config.num_speakers, config.speaker_embed_dim)
        
    def forward(self, audio=None, visual=None, text=None, speaker_ids=None):
        features = {}
        
        if audio is not None:
            features['audio'] = self.audio_encoder(audio)
        
        if visual is not None:
            features['visual'] = self.visual_encoder(visual)
            
        if text is not None:
            # Assuming text is already tokenized
            text_outputs = self.text_encoder(**text)
            text_features = text_outputs.last_hidden_state.mean(dim=1)  # Pool over sequence
            features['text'] = self.text_projection(text_features)
        
        if speaker_ids is not None:
            features['speaker'] = self.speaker_embedding(speaker_ids)
            
        return features

class LLMKeywordExtractor(nn.Module):
    """Extract emotion keywords using LLM (Qwen-7B)"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # In practice, load Qwen-7B
        # For now, simulate with a learnable module
        self.keyword_generator = nn.Sequential(
            nn.Linear(config.text_hidden_dim + config.hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, config.text_hidden_dim)
        )
        
    def forward(self, text_features, prev_emotion_state=None):
        """
        Extract keywords based on text and previous emotional context
        
        Args:
            text_features: Text embeddings [B, D]
            prev_emotion_state: Previous DGODE hidden state [B, H]
        
        Returns:
            keyword_features: Keyword embeddings [B, D]
        """
        if prev_emotion_state is None:
            prev_emotion_state = torch.zeros(
                text_features.size(0), 
                self.config.hidden_dim,
                device=text_features.device
            )
        
        # Concatenate text and emotional context
        combined = torch.cat([text_features, prev_emotion_state], dim=-1)
        
        # Generate keyword features
        keyword_features = self.keyword_generator(combined)
        
        return keyword_features

class GraphODEFunc(nn.Module):
    """ODE function for DGODE"""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.linear1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Tanh()
        
    def forward(self, t, h, adj=None):
        """
        Args:
            t: Time (scalar)
            h: Node features [N, D]
            adj: Adjacency matrix [N, N]
        """
        if adj is not None:
            # Graph convolution
            h_neighbors = torch.matmul(adj, h)
            h_combined = torch.cat([h, h_neighbors], dim=-1)
        else:
            h_combined = torch.cat([h, h], dim=-1)
            
        dh = self.linear2(self.activation(self.linear1(h_combined)))
        return dh

class DGODE(nn.Module):
    """Dialogue Graph ODE module"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Input projection
        input_dim = (config.audio_hidden_dim + config.visual_hidden_dim + 
                    config.text_hidden_dim + config.speaker_embed_dim)
        self.input_projection = nn.Linear(input_dim, config.hidden_dim)
        
        # ODE function
        self.ode_func = GraphODEFunc(config.hidden_dim)
        
        # Time parameters
        self.register_buffer('t', torch.tensor([0., 1.]))
        
    def build_graph(self, batch_size, speaker_ids, modality_masks):
        """Build adaptive conversation graph"""
        device = speaker_ids.device
        adj = torch.zeros(batch_size, batch_size, device=device)
        
        # Parameters for edge weights
        alpha1 = 0.8  # Same speaker weight
        alpha2 = 0.5  # Different speaker weight  
        beta = 0.1    # Temporal decay
        
        for i in range(batch_size):
            for j in range(batch_size):
                if i == j:
                    adj[i, j] = 1.0
                else:
                    time_diff = abs(i - j)
                    temporal_weight = torch.exp(-beta * time_diff)
                    
                    if speaker_ids[i] == speaker_ids[j]:
                        adj[i, j] = alpha1 * temporal_weight
                    else:
                        # Modality similarity factor
                        modality_sim = 1 - torch.abs(modality_masks[i] - modality_masks[j]).mean()
                        adj[i, j] = alpha2 * temporal_weight * modality_sim
        
        # Normalize adjacency matrix
        row_sum = adj.sum(dim=-1, keepdim=True) + 1e-8
        adj = adj / row_sum
        
        return adj
    
    def forward(self, features, speaker_ids, modality_masks):
        """
        Args:
            features: Concatenated features [B, D]
            speaker_ids: Speaker IDs [B]
            modality_masks: Modality availability [B, 3]
        """
        batch_size = features.size(0)
        
        # Project to hidden dimension
        h0 = self.input_projection(features)
        
        # Build adaptive graph
        adj = self.build_graph(batch_size, speaker_ids, modality_masks)
        
        # Solve ODE
        self.ode_func.adj = adj
        h_evolved = odeint(
            lambda t, h: self.ode_func(t, h, adj),
            h0,
            self.t,
            method='dopri5'
        )
        
        # Return final state
        return h_evolved[-1]

class ImaginationModule(nn.Module):
    """MMIN-style imagination module for missing modality reconstruction"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Define input dimensions for each modality reconstruction
        # Input: available modalities + keywords + context + speaker
        other_dim = (config.text_hidden_dim + config.hidden_dim + 
                    config.speaker_embed_dim)
        
        # Audio reconstruction
        self.audio_imagination = nn.Sequential(
            nn.Linear(config.visual_hidden_dim + other_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, config.audio_hidden_dim)
        )
        
        # Visual reconstruction  
        self.visual_imagination = nn.Sequential(
            nn.Linear(config.audio_hidden_dim + other_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, config.visual_hidden_dim)
        )
        
        # Text reconstruction
        self.text_imagination = nn.Sequential(
            nn.Linear(config.audio_hidden_dim + config.visual_hidden_dim + 
                     config.hidden_dim + config.speaker_embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, config.text_hidden_dim)
        )
        
    def forward(self, features, keywords, context, speaker_embed, modality_masks):
        """
        Reconstruct missing modalities
        
        Args:
            features: Dict of available features
            keywords: Keyword features [B, D]
            context: DGODE context [B, H]
            speaker_embed: Speaker embeddings [B, S]
            modality_masks: [B, 3] indicating availability (1=available, 0=missing)
        
        Returns:
            reconstructed: Dict of reconstructed features
        """
        reconstructed = {}
        batch_size = modality_masks.size(0)
        
        # Check each modality
        for i in range(batch_size):
            sample_recon = {}
            
            # Audio missing
            if modality_masks[i, 0] == 0:
                available = []
                if 'visual' in features and modality_masks[i, 1] == 1:
                    available.append(features['visual'][i])
                if 'text' in features and modality_masks[i, 2] == 1:
                    available.append(features['text'][i])
                available.extend([keywords[i], context[i], speaker_embed[i]])
                
                if available:
                    input_feat = torch.cat(available, dim=-1)
                    sample_recon['audio'] = self.audio_imagination(input_feat)
            
            # Visual missing
            if modality_masks[i, 1] == 0:
                available = []
                if 'audio' in features and modality_masks[i, 0] == 1:
                    available.append(features['audio'][i])
                if 'text' in features and modality_masks[i, 2] == 1:
                    available.append(features['text'][i])
                available.extend([keywords[i], context[i], speaker_embed[i]])
                
                if available:
                    input_feat = torch.cat(available, dim=-1)
                    sample_recon['visual'] = self.visual_imagination(input_feat)
            
            # Text missing
            if modality_masks[i, 2] == 0:
                available = []
                if 'audio' in features and modality_masks[i, 0] == 1:
                    available.append(features['audio'][i])
                if 'visual' in features and modality_masks[i, 1] == 1:
                    available.append(features['visual'][i])
                available.extend([context[i], speaker_embed[i]])
                
                if available:
                    input_feat = torch.cat(available, dim=-1)
                    sample_recon['text'] = self.text_imagination(input_feat)
            
            # Store reconstructed features
            for mod, feat in sample_recon.items():
                if mod not in reconstructed:
                    reconstructed[mod] = []
                reconstructed[mod].append(feat)
        
        # Stack reconstructed features
        for mod in reconstructed:
            reconstructed[mod] = torch.stack(reconstructed[mod], dim=0)
            
        return reconstructed

class AudioCLIPAlignment(nn.Module):
    """AudioCLIP alignment module for semantic consistency"""
    
    def __init__(self, config):
        super().__init__()
        # Simulate AudioCLIP encoders
        self.audio_encoder = nn.Linear(config.audio_hidden_dim, 256)
        self.text_encoder = nn.Linear(config.text_hidden_dim, 256)
        
    def forward(self, audio_features, text_features):
        """Calculate alignment loss between audio and text"""
        audio_embed = F.normalize(self.audio_encoder(audio_features), dim=-1)
        text_embed = F.normalize(self.text_encoder(text_features), dim=-1)
        
        # L1 loss as in the paper
        alignment_loss = F.l1_loss(audio_embed, text_embed)
        return alignment_loss

class MMIN_DGODE(nn.Module):
    """Main model combining MMIN, DGODE, and LLM guidance"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Feature extraction
        self.feature_extractor = FeatureExtractor(config)
        
        # LLM keyword extraction
        self.keyword_extractor = LLMKeywordExtractor(config)
        
        # DGODE for context
        self.dgode = DGODE(config)
        
        # Imagination module
        self.imagination = ImaginationModule(config)
        
        # AudioCLIP alignment
        self.audioclip_align = AudioCLIPAlignment(config)
        
        # Final classifier with skip connection
        final_dim = (config.audio_hidden_dim + config.visual_hidden_dim + 
                    config.text_hidden_dim + config.text_hidden_dim)  # +keywords
        
        self.classifier = nn.Sequential(
            nn.Linear(final_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.output_layer = nn.Linear(config.hidden_dim, config.num_classes)
        self.skip_connection = nn.Linear(final_dim, config.num_classes)
        
    def forward(self, audio=None, visual=None, text=None, speaker_ids=None,
                utterance_ids=None, modality_masks=None, labels=None, 
                return_losses=False):
        """
        Forward pass
        
        Args:
            audio, visual, text: Input modalities (can be None if missing)
            speaker_ids: Speaker IDs [B]
            utterance_ids: Utterance order [B]
            modality_masks: [B, 3] for (audio, visual, text) availability
            labels: Ground truth labels for consistency loss [B]
            return_losses: Whether to return individual loss components
        """
        batch_size = speaker_ids.size(0)
        device = speaker_ids.device
        
        # Extract features
        features = self.feature_extractor(audio, visual, text, speaker_ids)
        
        # Extract keywords (using text if available)
        if 'text' in features:
            keywords = self.keyword_extractor(features['text'])
        else:
            keywords = torch.zeros(batch_size, self.config.text_hidden_dim, device=device)
        
        # Prepare features for DGODE
        dgode_input = []
        for i in range(batch_size):
            sample_feat = []
            if modality_masks[i, 0] == 1 and 'audio' in features:
                sample_feat.append(features['audio'][i])
            else:
                sample_feat.append(torch.zeros(self.config.audio_hidden_dim, device=device))
                
            if modality_masks[i, 1] == 1 and 'visual' in features:
                sample_feat.append(features['visual'][i])
            else:
                sample_feat.append(torch.zeros(self.config.visual_hidden_dim, device=device))
                
            if modality_masks[i, 2] == 1 and 'text' in features:
                sample_feat.append(features['text'][i])
            else:
                sample_feat.append(torch.zeros(self.config.text_hidden_dim, device=device))
            
            sample_feat.append(features['speaker'][i])
            dgode_input.append(torch.cat(sample_feat, dim=-1))
        
        dgode_input = torch.stack(dgode_input, dim=0)
        
        # Get DGODE context
        context = self.dgode(dgode_input, speaker_ids, modality_masks)
        
        # Reconstruct missing modalities
        reconstructed = self.imagination(
            features, keywords, context, 
            features['speaker'], modality_masks
        )
        
        # Combine original and reconstructed features
        final_features = {}
        for mod in ['audio', 'visual', 'text']:
            if mod in features:
                final_features[mod] = features[mod]
            if mod in reconstructed:
                # Use reconstructed for missing samples
                if mod not in final_features:
                    final_features[mod] = reconstructed[mod]
                else:
                    # Combine based on masks
                    mod_idx = {'audio': 0, 'visual': 1, 'text': 2}[mod]
                    for i in range(batch_size):
                        if modality_masks[i, mod_idx] == 0:
                            final_features[mod][i] = reconstructed[mod][i]
        
        # Prepare final representation
        final_repr = []
        for i in range(batch_size):
            sample_repr = []
            for mod in ['audio', 'visual', 'text']:
                if mod in final_features:
                    sample_repr.append(final_features[mod][i])
                else:
                    dim = getattr(self.config, f'{mod}_hidden_dim')
                    sample_repr.append(torch.zeros(dim, device=device))
            sample_repr.append(keywords[i])
            final_repr.append(torch.cat(sample_repr, dim=-1))
        
        final_repr = torch.stack(final_repr, dim=0)
        
        # Classification with skip connection
        hidden = self.classifier(final_repr)
        logits = self.output_layer(hidden) + self.skip_connection(final_repr)
        
        outputs = {'logits': logits}
        
        # Calculate losses if needed
        if return_losses and labels is not None:
            # Classification loss
            cls_loss = F.cross_entropy(logits, labels)
            outputs['cls_loss'] = cls_loss
            
            # Reconstruction loss
            recon_loss = 0
            count = 0
            for mod in ['audio', 'visual', 'text']:
                if mod in features and mod in reconstructed:
                    mod_idx = {'audio': 0, 'visual': 1, 'text': 2}[mod]
                    for i in range(batch_size):
                        if modality_masks[i, mod_idx] == 0:
                            recon_loss += F.l1_loss(
                                reconstructed[mod][i],
                                features[mod][i]
                            )
                            count += 1
            
            if count > 0:
                outputs['recon_loss'] = recon_loss / count
            
            # Alignment loss (AudioCLIP)
            if 'audio' in reconstructed and 'text' in final_features:
                align_loss = self.audioclip_align(
                    reconstructed['audio'],
                    final_features['text']
                )
                outputs['align_loss'] = align_loss
            
            # Consistency loss - CORRECTED VERSION
            if labels is not None:
                consist_loss = 0
                count = 0
                for i in range(batch_size):
                    for j in range(i+1, batch_size):
                        if labels[i] == labels[j]:
                            # Same emotion - should have similar representations
                            similarity = F.cosine_similarity(
                                final_repr[i].unsqueeze(0),
                                final_repr[j].unsqueeze(0)
                            )
                            # Loss = 1 - similarity (minimize when similar)
                            consist_loss += (1 - similarity)
                            count += 1
                
                if count > 0:
                    outputs['consist_loss'] = consist_loss / count
        
        return outputs
