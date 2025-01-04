# coding:utf8
import sys, os
import torch as t
from data import get_data
from model import PoetryModel
from torch import nn
from utils import Visualizer
import tqdm
from torchnet import meter
import ipdb
import time
import warnings


class Config(object):
    data_path = "data/"
    pickle_path = "./data/tang.npz"
    author = None  # 只学习某位作者的诗歌
    constrain = None  # 长度限制
    category = "poet.tang"  # 类别，唐诗还是宋诗歌(poet.song)
    lr = 1e-3
    weight_decay = 1e-4
    use_gpu = (
        True  # Default to False, but will be auto-disabled if CUDA is not available
    )
    epoch = 20
    batch_size = 128
    maxlen = 125  # 超过这个长度的之后字被丢弃，小于这个长度的在前面补空格
    plot_every = 20  # 每20个batch 可视化一次
    # use_env = True # 是否使用visodm
    env = "poetry"  # visdom env
    max_gen_len = 200  # 生成诗歌最长长度
    debug_file = "/tmp/debugp"
    model_path = None  # 预训练模型路径
    prefix_words = (
        "细雨鱼儿出,微风燕子斜。"  # 不是诗歌的组成部分，用来控制生成诗歌的意境
    )
    start_words = "闲云潭影日悠悠"  # 诗歌开始
    acrostic = False  # 是否是藏头诗
    model_prefix = "checkpoints/tang"  # 模型保存路径
    use_visdom = False  # Default visualization option
    env = "poetry"  # visdom env if used


opt = Config()


def generate(model, start_words, ix2word, word2ix, prefix_words=None):
    """
    给定几个词，根据这几个词接着生成一首完整的诗歌
    start_words：u'春江潮水连海平'
    比如start_words 为 春江潮水连海平，可以生成：

    """

    results = list(start_words)
    start_word_len = len(start_words)
    # 手动设置第一个词为<START>
    input = t.Tensor([word2ix["<START>"]]).view(1, 1).long()
    if opt.use_gpu:
        input = input.cuda()
    hidden = None

    if prefix_words:
        for word in prefix_words:
            output, hidden = model(input, hidden)
            input = input.data.new([word2ix[word]]).view(1, 1)

    for i in range(opt.max_gen_len):
        output, hidden = model(input, hidden)

        if i < start_word_len:
            w = results[i]
            input = input.data.new([word2ix[w]]).view(1, 1)
        else:
            top_index = output.data[0].topk(1)[1][0].item()
            w = ix2word[top_index]
            results.append(w)
            input = input.data.new([top_index]).view(1, 1)
        if w == "<EOP>":
            del results[-1]
            break
    return results


def gen_acrostic(model, start_words, ix2word, word2ix, prefix_words=None):
    """
    生成藏头诗
    start_words : u'深度学习'
    生成：
    深木通中岳，青苔半日脂。
    度山分地险，逆浪到南巴。
    学道兵犹毒，当时燕不移。
    习根通古岸，开镜出清羸。
    """
    results = []
    start_word_len = len(start_words)
    input = t.Tensor([word2ix["<START>"]]).view(1, 1).long()
    if opt.use_gpu:
        input = input.cuda()
    hidden = None

    index = 0  # 用来指示已经生成了多少句藏头诗
    # 上一个词
    pre_word = "<START>"

    if prefix_words:
        for word in prefix_words:
            output, hidden = model(input, hidden)
            input = (input.data.new([word2ix[word]])).view(1, 1)

    for i in range(opt.max_gen_len):
        output, hidden = model(input, hidden)
        top_index = output.data[0].topk(1)[1][0].item()
        w = ix2word[top_index]

        if pre_word in {"。", "！", "<START>"}:
            # 如果遇到句号，藏头的词送进去生成

            if index == start_word_len:
                # 如果生成的诗歌已经包含全部藏头的词，则结束
                break
            else:
                # 把藏头的词作为输入送入模型
                w = start_words[index]
                index += 1
                input = (input.data.new([word2ix[w]])).view(1, 1)
        else:
            # 否则的话，把上一次预测是词作为下一个词输入
            input = (input.data.new([word2ix[w]])).view(1, 1)
        results.append(w)
        pre_word = w
    return results


def train(**kwargs):
    for k, v in kwargs.items():
        setattr(opt, k, v)

    # Check CUDA availability and update use_gpu setting
    if opt.use_gpu and not t.cuda.is_available():
        print("Warning: CUDA not available, using CPU instead")
        opt.use_gpu = False

    opt.device = t.device("cuda") if opt.use_gpu else t.device("cpu")
    device = opt.device
    print(f"Using device: {device}")

    # Modify the Visualizer initialization to be conditional
    vis = Visualizer(env=opt.env) if opt.use_visdom else None

    # 获取数据
    data, word2ix, ix2word = get_data(opt)
    print(f"Data shape after loading: {data.shape}")
    data = t.from_numpy(data)
    
    # Move model to device before creating optimizer
    model = PoetryModel(len(word2ix), 128, 256)
    model.to(device)
    
    dataloader = t.utils.data.DataLoader(
        data,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=1
    )

    # Estimate training time before starting
    print("\nEstimating training time...")
    total_batches = len(dataloader)
    
    # Time multiple batches for a more accurate estimate
    num_test_batches = min(5, total_batches)  # Test with 5 batches or all if less
    test_times = []
    
    # Get iterator once
    dataloader_iter = iter(dataloader)
    
    for _ in range(num_test_batches):
        try:
            start_time = time.time()
            
            # Get batch and process it
            sample_batch = next(dataloader_iter)
            sample_batch = sample_batch.long().transpose(1, 0).contiguous().to(device)
            
            with t.no_grad():
                input_, target = sample_batch[:-1, :], sample_batch[1:, :]
                output, _ = model(input_)
                
            batch_time = time.time() - start_time
            test_times.append(batch_time)
            
        except StopIteration:
            break
    
    # # Calculate average batch time
    # avg_batch_time = sum(test_times) / len(test_times)
    
    # # Calculate estimates
    # estimated_epoch_time = avg_batch_time * total_batches
    # estimated_total_time = estimated_epoch_time * opt.epoch
    
    # print(f"\nTraining Time Estimates (based on {len(test_times)} test batches):")
    # print(f"Average time per batch: {avg_batch_time:.2f} seconds")
    # print(f"Estimated time per epoch: {estimated_epoch_time/60:.2f} minutes")
    # print(f"Estimated total training time: {estimated_total_time/3600:.2f} hours")
    # print(f"Number of batches per epoch: {total_batches}")
    # print(f"Total number of epochs: {opt.epoch}")
    
    # user_input = input("\nProceed with training? (y/n): ")
    # if user_input.lower() != 'y':
    #     print("Training cancelled")
    #     return

    optimizer = t.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.CrossEntropyLoss()
    if opt.model_path:
        state_dict = t.load(opt.model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)

    loss_meter = meter.AverageValueMeter()
    total_start_time = time.time()  # Track total training time
    
    for epoch in range(opt.epoch):
        epoch_start_time = time.time()  # Track epoch time
        loss_meter.reset()
        print(f"\nEpoch {epoch+1}/{opt.epoch} - Total batches: {len(dataloader)}")
        
        for ii, data_ in tqdm.tqdm(enumerate(dataloader)):
            # Move data to device immediately after loading
            data_ = data_.long().transpose(1, 0).contiguous().to(device)
            
            data_ = data_.long().transpose(1, 0).contiguous()
            # After transpose: shape becomes [maxlen, batch_size] = [125, 128]
            # print(f"After transpose data_ shape: {data_.shape}")  # Should be [125, 128]
            
            # 移动数据到GPU/CPU
            data_ = data_.to(device)
            
            # 清空梯度
            optimizer.zero_grad()
            
            # 准备输入和目标
            input_, target = data_[:-1, :], data_[1:, :]
            # print(f"input_ shape: {input_.shape}")   # Should be [124, 128]
            # print(f"target shape: {target.shape}")  # Should be [124, 128]
            
            # 前向传播
            output, _ = model(input_)
            # print(f"output shape: {output.shape}")   # Should be [124 * 128, vocab_size]

            # target.view(-1) flattens the target tensor from [124, 128] to [15872] (124*128)
            # This matches output shape [15872, vocab_size] for CrossEntropyLoss
            loss = criterion(output, target.view(-1))
            loss.backward()
            optimizer.step()

            loss_meter.add(loss.item())

            # Modify visualization code to be conditional
            if opt.use_visdom and (1 + ii) % opt.plot_every == 0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                vis.plot("loss", loss_meter.value()[0])

                # 诗歌原文
                poetrys = [
                    [ix2word[_word] for _word in data_[:, _iii].tolist()]
                    for _iii in range(data_.shape[1])
                ][:16]
                vis.text(
                    "</br>".join(["".join(poetry) for poetry in poetrys]),
                    win="origin_poem",
                )

                gen_poetries = []
                # 分别以这几个字作为诗歌的第一个字，生成8首诗
                for word in list("春江花月夜凉如水"):
                    gen_poetry = "".join(generate(model, word, ix2word, word2ix))
                    gen_poetries.append(gen_poetry)
                vis.text(
                    "</br>".join(["".join(poetry) for poetry in gen_poetries]),
                    win="gen_poem",
                )

            # Add non-visdom progress reporting
            elif (1 + ii) % opt.plot_every == 0:
                print(f"\nProgress: Batch {ii+1}/{len(dataloader)} ({(ii+1)/len(dataloader)*100:.1f}%)")
                print(f"Current Loss: {loss_meter.value()[0]:.4f}")

        # Calculate and print timing information after each epoch
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - total_start_time
        remaining_epochs = opt.epoch - (epoch + 1)
        estimated_remaining_time = (total_time / (epoch + 1)) * remaining_epochs
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Time for this epoch: {epoch_time:.2f} seconds")
        print(f"Average epoch time: {total_time/(epoch+1):.2f} seconds")
        print(f"Estimated remaining time: {estimated_remaining_time/60:.2f} minutes")
        print(f"Total time elapsed: {total_time/60:.2f} minutes")
        
        # Save model
        t.save(model.state_dict(), "%s_%s.pth" % (opt.model_prefix, epoch))


def gen(**kwargs):
    """
    提供命令行接口，用以生成相应的诗
    """
    for k, v in kwargs.items():
        setattr(opt, k, v)

    # Check CUDA availability and update use_gpu setting
    if opt.use_gpu and not t.cuda.is_available():
        print("Warning: CUDA not available, using CPU instead")
        opt.use_gpu = False

    # First load the data to get vocabulary size
    data, word2ix, ix2word = get_data(opt)
    vocab_size = len(word2ix)
    print(f"Vocabulary size: {vocab_size}")
    
    # Print some example characters from the vocabulary
    print("\nExample characters in vocabulary:")
    sample_chars = list(word2ix.keys())[:20]  # First 20 characters
    print(''.join(sample_chars))
    
    # Check if input characters are in vocabulary
    invalid_chars = [char for char in opt.start_words if char not in word2ix]
    if invalid_chars:
        print(f"\nWarning: The following characters are not in the vocabulary: {''.join(invalid_chars)}")
        print("Please use only classical Chinese characters that appear in Tang poetry.")
        return
    
    # Then create the model with correct vocabulary size
    model = PoetryModel(len(word2ix), 128, 256)
    
    # Load the pretrained weights with weights_only=True to avoid the warning
    map_location = "cuda" if opt.use_gpu else "cpu"
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            state_dict = t.load(opt.model_path, map_location=map_location, weights_only=True)
        model.load_state_dict(state_dict)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    if opt.use_gpu:
        model.cuda()

    # python2和python3 字符串兼容
    if sys.version_info.major == 3:
        if opt.start_words.isprintable():
            start_words = opt.start_words
            prefix_words = opt.prefix_words if opt.prefix_words else None
        else:
            start_words = opt.start_words.encode("ascii", "surrogateescape").decode("utf8")
            prefix_words = (
                opt.prefix_words.encode("ascii", "surrogateescape").decode("utf8")
                if opt.prefix_words
                else None
            )
    else:
        start_words = opt.start_words.decode("utf8")
        prefix_words = opt.prefix_words.decode("utf8") if opt.prefix_words else None

    start_words = start_words.replace(",", "，").replace(".", "。").replace("?", "？")

    gen_poetry = gen_acrostic if opt.acrostic else generate
    result = gen_poetry(model, start_words, ix2word, word2ix, prefix_words)
    print("".join(result))


if __name__ == "__main__":
    import fire

    fire.Fire()
