use burn::{
    config::Config,
    data::{
        dataloader::{batcher::Batcher, DataLoaderBuilder},
        dataset::InMemDataset,
    },
    module::Module,
    optim::AdamConfig,
    prelude::Backend,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        metric::{AccuracyMetric, LossMetric},
        LearnerBuilder, RegressionOutput, TrainOutput, TrainStep, ValidStep,
    },
};
use rand::{rngs::SmallRng, Rng, SeedableRng};

use crate::{
    ai::{
        batcher::{MinesweeperBatch, MinesweeperBatcher},
        model::{Model, ModelConfig},
    },
    game::{Minesweeper, Square},
};

impl<B: AutodiffBackend> TrainStep<MinesweeperBatch<B>, RegressionOutput<B>> for Model<B> {
    fn step(&self, batch: MinesweeperBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forward_regression(batch.boards, batch.mines);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<MinesweeperBatch<B>, RegressionOutput<B>> for Model<B> {
    fn step(&self, batch: MinesweeperBatch<B>) -> RegressionOutput<B> {
        self.forward_regression(batch.boards, batch.mines)
    }
}

#[derive(Config)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 100)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(config.seed);

    let batcher = MinesweeperBatcher::default();

    println!("Generating training and test data...");
    let train_data = InMemDataset::new(generate_data(10_000, config.seed));
    let test_data = InMemDataset::new(generate_data(1_000, config.seed));
    println!("Done generating data");

    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train_data);

    let dataloader_test = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(test_data);

    let learner = LearnerBuilder::new(artifact_dir)
        // .metric_train_numeric(AccuracyMetric::new())
        // .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            config.model.init::<B>(&device),
            config.optimizer.init(),
            config.learning_rate,
        );

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}

fn generate_data(amount: u32, seed: u64) -> Vec<Minesweeper> {
    let mut rng = SmallRng::seed_from_u64(seed);

    (0..amount)
        .map(|_| {
            let mines = rng.random_range(4..100);
            let mut game = Minesweeper::new_with_mines(mines);

            while !game.is_board_completed() {
                let row = rng.random_range(0..game.grid.len());
                let col = rng.random_range(0..game.grid[0].len());

                let clicked_square = game.click(row, col);
                if matches!(clicked_square, Square::Mine) {
                    break;
                }
            }

            game
        })
        .collect()
}
