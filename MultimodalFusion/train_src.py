from trainer import FusionTrainer
from draw_graph import plot


if __name__ == '__main__':
    num_epoch = 150
    model = FusionTrainer(
        model_name='src',
        device='cpu',
        init_lr=0.05
    )
    model.train(num_epoch)
    # plot('src', xlim=num_epoch)

