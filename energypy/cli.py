import click

from energypy.train import main as train_main



@click.command()
@click.argument('entry', nargs=1)
@click.argument('agent', nargs=1)
@click.argument('environment', nargs=1)
@click.argument('learner', nargs=1)
@click.option('-n_gen', '--n_generations', default=10, type=int)
@click.option('-n_col', '--n_collectors', default=2, type=int)
def main(
    entry,
    agent,
    learner,
    environment,
    n_generations,
    n_collectors
):
    if entry == 'train':
        train_main(
            agent,
            environment,
            learner,
            n_generations,
            n_collectors
        )
