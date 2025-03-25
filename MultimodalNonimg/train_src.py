from trainer import NonimgTrainer
from draw_graph import plot


if __name__ == '__main__':
    model = NonimgTrainer(
        model_name='src',
        device='cpu',
        init_lr=0.01
    )
    model.train()
    plot('src')

