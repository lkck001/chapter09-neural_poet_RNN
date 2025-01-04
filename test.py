import torch as t
from data import get_data
from main import Config, PoetryModel
import numpy as np

def test_data_loading():
    """Test data loading and examine data structures"""
    # Initialize config
    opt = Config()
    opt.pickle_path = './data/tang.npz'
    
    # Get data
    data, word2ix, ix2word = get_data(opt)
    print("\n=== Data Overview ===")
    print(f"Data shape: {data.shape}")
    print(f"Vocabulary size: {len(word2ix)}")
    
    # Check special tokens and handle potential missing tokens
    print("\n=== Special Tokens ===")
    special_tokens = ['<START>', '<EOP>', ' ']
    for token in special_tokens:
        try:
            print(f"{token}: {word2ix.get(token, 'Not found')}")
        except:
            print(f"{token}: Not in vocabulary")
    
    # Show some sample vocabulary items
    print("\n=== Sample Characters ===")
    sample_chars = list("春江花月夜")
    for char in sample_chars:
        if char in word2ix:
            print(f"{char}: {word2ix[char]}")
    
    # Show first few items in ix2word
    print("\n=== First Few ix2word Mappings ===")
    for i in range(min(10, len(ix2word))):
        print(f"{i}: {ix2word[i]}")
    
    return data, word2ix, ix2word, opt

def test_dataloader(data, opt, ix2word):
    """Test DataLoader functionality"""
    print("\n=== DataLoader Test ===")
    data = t.from_numpy(data)
    dataloader = t.utils.data.DataLoader(
        data,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=1
    )
    
    # Get one batch
    batch = next(iter(dataloader))
    print(f"Batch shape: {batch.shape}")
    
    # Show contents of first few poems in the batch
    print("\n=== Sample Poems from Batch ===")
    for poem_idx in range(min(3, batch.shape[0])):  # Show first 3 poems
        poem = batch[poem_idx]
        # Convert indices to characters and filter out padding
        chars = [ix2word[idx.item()] for idx in poem if idx.item() < len(ix2word)]
        print(f"\nPoem {poem_idx + 1}:")
        print(''.join(chars))
        print(f"Length: {len(chars)} characters")
        
        # Show raw indices for debugging
        print("First 10 indices:", poem[:10].tolist())
    
    return dataloader

def test_model(word2ix, opt):
    """Test model initialization and basic forward pass"""
    print("\n=== Model Test ===")
    
    # Initialize model
    model = PoetryModel(len(word2ix), 128, 256)
    print(f"Model vocabulary size: {len(word2ix)}")
    
    # Test with a small batch
    batch_size = 2
    seq_len = 10
    input_data = t.randint(0, len(word2ix), (seq_len, batch_size))
    
    # Forward pass
    output, hidden = model(input_data)
    print(f"Input shape: {input_data.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Hidden shape: {hidden[0].shape}")
    
    return model

def test_poem_generation(model, word2ix, ix2word):
    """Test basic poem generation"""
    print("\n=== Poem Generation Test ===")
    
    # Set model to eval mode
    model.eval()
    
    # Start token
    start_token = t.tensor([word2ix['<START>']]).view(1, 1)
    
    # Generate sequence
    max_len = 20
    generated = []
    hidden = None
    
    with t.no_grad():
        for i in range(max_len):
            output, hidden = model(start_token, hidden)
            
            # Get next word
            top_idx = output.argmax(dim=1)
            next_word = ix2word[top_idx.item()]
            
            if next_word == '<EOP>':
                break
                
            generated.append(next_word)
            start_token = top_idx.view(1, 1)
    
    print("Generated sequence:")
    print(''.join(generated))

def main():
    # Run all tests
    data, word2ix, ix2word, opt = test_data_loading()
    dataloader = test_dataloader(data, opt, ix2word)
    model = test_model(word2ix, opt)
    test_poem_generation(model, word2ix, ix2word)

if __name__ == "__main__":
    main() 