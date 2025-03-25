from trainer import ARP2DCNNTrainer
from compute_performance import print_info
from draw_graph import plot


if __name__ == '__main__':
    num_epoch = 100
    model_name = 'src'
    model = ARP2DCNNTrainer(
        model_name=model_name,
        device='cuda:0',
        batch_size=16,
        init_lr=0.0001
    )
    model.train(num_epoch=num_epoch)
    print_info([model_name])
    plot(model_name)
