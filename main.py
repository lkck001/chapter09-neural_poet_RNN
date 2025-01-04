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
    print(f"Data shape after loading: {data.shape}")  # Check if data is empty
    data = t.from_numpy(data)
    # data shape: [total_poems, max_length]
    # - total_poems: number of poems in the dataset
    # - max_length: maximum length of poems (padded with zeros)
    # Each element is an integer index corresponding to a word in the vocabulary
    
    dataloader = t.utils.data.DataLoader(
        data, 
        batch_size=opt.batch_size,  # Creates batches of opt.batch_size poems
        shuffle=True,  # Randomly shuffles poems during training
        num_workers=1   # Number of parallel workers for data loading
    )
    # dataloader yields tensors of shape [batch_size, maxlen]
    # where maxlen = 125 (from Config)

    # 模型定义
    model = PoetryModel(len(word2ix), 128, 256)
    # Parameters:
    # 1. len(word2ix): vocabulary size - number of unique characters/words in the dataset
    # 2. 128: embedding dimension - size of the vector space where words are embedded
    # 3. 256: hidden dimension - size of the hidden state in the RNN/LSTM

    # Estimate training time before starting
    print("\nEstimating training time...")
    total_batches = len(dataloader)
    sample_batch = next(iter(dataloader))
    sample_batch = sample_batch.long().transpose(1, 0).contiguous().to(device)
    
    # Time one batch
    start_time = time.time()
    with t.no_grad():
        input_, target = sample_batch[:-1, :], sample_batch[1:, :]
        output, _ = model(input_)
    batch_time = time.time() - start_time
    
    # Calculate estimates
    estimated_epoch_time = batch_time * total_batches
    estimated_total_time = estimated_epoch_time * opt.epoch
    
    print(f"\nTraining Time Estimates:")
    print(f"Time per batch: {batch_time:.2f} seconds")
    print(f"Estimated time per epoch: {estimated_epoch_time/60:.2f} minutes")
    print(f"Estimated total training time: {estimated_total_time/3600:.2f} hours")
    print(f"Number of batches per epoch: {total_batches}")
    print(f"Total number of epochs: {opt.epoch}")
    
    user_input = input("\nProceed with training? (y/n): ")
    if user_input.lower() != 'y':
        print("Training cancelled")
        return

    optimizer = t.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.CrossEntropyLoss()
    if opt.model_path:
        model.load_state_dict(t.load(opt.model_path))
    model.to(device)

    loss_meter = meter.AverageValueMeter()
    for epoch in range(opt.epoch):
        loss_meter.reset()
        for ii, data_ in tqdm.tqdm(enumerate(dataloader)):
            # data_ shape: [batch_size, maxlen] = [128, 125]
            print(f"\nOriginal data_ shape: {data_.shape}")  # Should be [128, 125]
            
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
            print(f"output shape: {output.shape}")   # Should be [124 * 128, vocab_size]

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
                print(f"Epoch: {epoch}, Batch: {ii}, Loss: {loss_meter.value()[0]:.4f}")

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

    data, word2ix, ix2word = get_data(opt)
    model = PoetryModel(len(word2ix), 128, 256)
    map_location = "cuda" if opt.use_gpu else "cpu"
    state_dict = t.load(opt.model_path, map_location=map_location)
    model.load_state_dict(state_dict)

    if opt.use_gpu:
        model.cuda()

    # python2和python3 字符串兼容
    if sys.version_info.major == 3:
        if opt.start_words.isprintable():
            start_words = opt.start_words
            prefix_words = opt.prefix_words if opt.prefix_words else None
        else:
            start_words = opt.start_words.encode("ascii", "surrogateescape").decode(
                "utf8"
            )
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
