import os
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from pydub import AudioSegment

device = "cuda" if torch.cuda.is_available() else "cpu"

# enhanced MFCC feature extraction
def loader_improved(path):
    audio, sample_rate = torchaudio.load(path)
    audio = torchaudio.functional.resample(audio, orig_freq=sample_rate, new_freq=16000)
    audio = audio.mean(dim=0, keepdim=True)
    
    # Increase the number of MFCCs to 40 (instead of 13)
    mfcc_features = torchaudio.transforms.MFCC(
        sample_rate=16000,
        n_mfcc=40,
        melkwargs={
            "n_mels": 80,
            "n_fft": 512,
            "hop_length": 160
        }
    )(audio)
    
    # Always keep the mean for this loader
    mfcc_features = mfcc_features.mean(dim=2).squeeze()
    return mfcc_features

# MFCC sequence extraction (for RNN approach)
def sequence_loader(path, max_length=300):
    audio, sample_rate = torchaudio.load(path)
    audio = torchaudio.functional.resample(audio, orig_freq=sample_rate, new_freq=16000)
    audio = audio.mean(dim=0, keepdim=True)
    
    # Extract MFCC with enhanced parameters
    mfcc_features = torchaudio.transforms.MFCC(
        sample_rate=16000,
        n_mfcc=40,
        melkwargs={
            "n_mels": 80,
            "n_fft": 512,
            "hop_length": 160
        }
    )(audio)
    
    # Transpose to have shape [time, features]
    mfcc_features = mfcc_features.squeeze(0).transpose(0, 1)
    
    # Handle variable-length sequences
    seq_len = mfcc_features.shape[0]
    
    if seq_len > max_length:
        # If too long, truncate
        mfcc_features = mfcc_features[:max_length, :]
    elif seq_len < max_length:
        # If too short, pad with zeros
        padding = torch.zeros(max_length - seq_len, mfcc_features.shape[1])
        mfcc_features = torch.cat([mfcc_features, padding], dim=0)
    
    return mfcc_features, min(seq_len, max_length)

# custom dataset for sequences
class SequenceSpeakerDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_length=300):
        self.root_dir = root_dir
        self.transform = transform
        self.max_length = max_length
        
        self.samples = []
        self.class_to_idx = {}
        
        # Retrieve classes (speaker names)
        classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) 
                   and not d.startswith('.') and not d.startswith('_')]
        
        # Create class -> index mapping
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        self.classes = classes
        
        # Collect all samples
        for cls_name in classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for filename in os.listdir(cls_dir):
                if filename.endswith('.wav'):
                    path = os.path.join(cls_dir, filename)
                    self.samples.append((path, self.class_to_idx[cls_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        mfcc_seq, length = sequence_loader(path, self.max_length)
        
        if self.transform:
            mfcc_seq = self.transform(mfcc_seq)
        
        return mfcc_seq, length, label

# collate function for DataLoader
def collate_fn(batch):
    data, lengths, labels = zip(*batch)
    data = torch.stack(data)
    lengths = torch.tensor(lengths)
    labels = torch.tensor(labels)
    return data, lengths, labels

# enhanced RNN model for speaker recognition
class SpeakerRecognitionRNN(nn.Module):
    def __init__(self, input_size=40, hidden_size=128, num_layers=2, num_classes=10, dropout=0.3):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x, lengths=None):
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, _ = self.lstm(x)
        if lengths is not None:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        
        # Use mean over time dimension
        out_avg = torch.mean(output, dim=1)
        return self.fc(out_avg)


# accuracy calculation function
def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc
            
# Train the model
def train_rnn_model(model, train_loader, val_loader, criterion, optimizer, n_epochs, device):
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        epoch_acc = 0
        
        for batch_idx, (data, lengths, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            lengths = lengths.to(device)
            
            optimizer.zero_grad()
            output = model(data, lengths)
            loss = criterion(output, target)
            acc = calculate_accuracy(output, target)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
        train_losses.append(epoch_loss / len(train_loader))
        train_accs.append(epoch_acc / len(train_loader))
        
        model.eval()
        epoch_loss = 0
        epoch_acc = 0
        
        with torch.no_grad():
            for data, lengths, target in val_loader:
                data, target = data.to(device), target.to(device)
                lengths = lengths.to(device)
                
                output = model(data, lengths)
                loss = criterion(output, target)
                acc = calculate_accuracy(output, target)
                
                epoch_loss += loss.item()
                epoch_acc += acc.item()
        
        val_losses.append(epoch_loss / len(val_loader))
        val_accs.append(epoch_acc / len(val_loader))
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch: {epoch+1:02}')
            print(f'\tTrain Loss: {train_losses[-1]:.3f} | Train Acc: {train_accs[-1]:.3f}')
            print(f'\t Val. Loss: {val_losses[-1]:.3f} |  Val. Acc: {val_accs[-1]:.3f}')
    
    return train_losses, train_accs, val_losses, val_accs

# function to predict the speaker
def predict_speaker(model, audio_path, class_names, device, max_length=300):
    model.eval()
    
    mfcc_seq, length = sequence_loader(audio_path, max_length)
    mfcc_seq = mfcc_seq.unsqueeze(0).to(device)
    length = torch.tensor([length]).to(device)
    
    with torch.no_grad():
        output = model(mfcc_seq, length)
        probs = torch.softmax(output, dim=1)
        
        top_probs, top_indices = torch.topk(probs, k=min(3, len(class_names)), dim=1)
        
        results = []
        for i in range(top_indices.size(1)):
            idx = top_indices[0, i].item()
            prob = top_probs[0, i].item() * 100
            speaker_name = class_names[idx]
            results.append((speaker_name, prob))
    
    return results

# See the embeddings of the model
def visualize_embeddings(model, dataset, device, method='pca'):
    model.eval()
    embeddings = []
    labels = []
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    # extract the embeddings
    with torch.no_grad():
        for data, lengths, target in dataloader:
            data, lengths = data.to(device), lengths.to(device)
            
            if isinstance(model, SpeakerRecognitionRNN):
                packed = nn.utils.rnn.pack_padded_sequence(data, lengths.cpu(), batch_first=True, enforce_sorted=False)
                _, (hidden, _) = model.lstm(packed)
                hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
                embeddings.append(hidden.cpu().numpy())
            else:
                pass
            
            labels.append(target.cpu().numpy())
    
    embeddings = np.vstack(embeddings)
    labels = np.concatenate(labels)
    
    # reduce dimensionnality
    if method.lower() == 'pca':
        reducer = PCA(n_components=2)
    else:
        reducer = TSNE(n_components=2, random_state=42)
    
    embeddings_2d = reducer.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    
    unique_labels = np.unique(labels)
    class_names = {i: name for name, i in dataset.class_to_idx.items()}
    
    for label in unique_labels:
        idx = labels == label
        plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=class_names[label])
    
    plt.title(f'View embeddings by {method.upper()}')
    plt.legend()
    plt.savefig(f'embeddings_{method.lower()}.png')
    
    return plt

# convert mp3 files to wav
def convert_mp3_to_wav(root_dir):
    def list_not_hidden(directory):
        return [f for f in os.listdir(directory) if not f.startswith(('.','_'))]

    for speaker in list_not_hidden(root_dir):
        speaker_dir = os.path.join(root_dir, speaker)
        print(f"Processing of -- {speaker_dir}")

        for f in list_not_hidden(speaker_dir):
            if f.endswith(".mp3"):
                mp3_path = os.path.join(speaker_dir, f)
                wav_path = os.path.join(speaker_dir, f"{os.path.splitext(f)[0]}.wav")
                if not os.path.isfile(wav_path):
                    AudioSegment.from_mp3(mp3_path).export(wav_path, format="wav")
                    print(f"Converting {mp3_path} vers {wav_path}")