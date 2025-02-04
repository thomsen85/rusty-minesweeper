use std::collections::VecDeque;

use burn::{
    backend::{Autodiff, Wgpu},
    module::AutodiffModule,
    nn::{
        loss::{CrossEntropyLoss, MseLoss},
        Dropout, DropoutConfig, Linear, LinearConfig, Relu,
    },
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::*,
    record::{CompactRecorder, Record},
    tensor::{backend::AutodiffBackend, ops::FloatElem},
};
use plotters::prelude::*;

use crate::constants;
use crate::game;
use rand::seq::{IndexedRandom, IteratorRandom};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    dropout: Dropout,
    activation: Relu,
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    in_classes: usize,
    hidden_size: usize,
    out_classes: usize,
    #[config(default = "0.5")]
    dropout: f64,
}

impl ModelConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            activation: Relu::new(),
            linear1: LinearConfig::new(self.in_classes, self.hidden_size).init(device),
            linear2: LinearConfig::new(self.hidden_size, self.out_classes).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}
pub type MyBackend = Wgpu<f32, i32>;
pub type MyAutodiffBackend = Autodiff<MyBackend>;

impl<B: Backend> Model<B> {
    /// # Shapes
    ///   - Images [batch_size, flattened_board]
    ///   - Output [batch_size, num_classes]
    pub fn forward(&self, board_states: Tensor<B, 1>) -> Tensor<B, 1> {
        // let [batch_size, flattened_board_size] = board_states.dims();

        // Create a channel at the second dimension.
        // let x = board_states.reshape([batch_size, 1, height, width]);

        // let x = self.conv1.forward(x); // [batch_size, 8, _, _]
        // let x = self.dropout.forward(x);
        // let x = self.conv2.forward(x); // [batch_size, 16, _, _]
        // let x = self.dropout.forward(x);
        // let x = self.activation.forward(x);
        //
        // let x = self.pool.forward(x); // [batch_size, 16, 8, 8]
        // let x = x.reshape([batch_size, 1]);
        let x = self.linear1.forward(board_states);
        // let x = self.dropout.forward(x);
        // let x = self.activation.forward(x);
        self.linear2.forward(x) // [batch_size, num_classes]}
    }
}

#[derive(Config)]
pub struct TrainingConfig {
    #[config(default = 2048)]
    pub replay_mem_capacity: usize,

    #[config(default = 1000)]
    pub num_episodes: usize,

    #[config(default = 1.0)]
    pub epsilon: f32,

    // #[config(default = 0.001)]
    // pub decay_rate: f32,
    #[config(default = 10)]
    pub sync_rate: usize,

    #[config(default = 128)]
    pub batch_size: usize,
    // #[config(default = 4)]
    // pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 5e-6)]
    pub lr: f64,
}

struct Memory {
    memory: VecDeque<Experience>,
    max_size: usize,
}

impl Memory {
    fn new(max_size: usize) -> Self {
        Self {
            memory: VecDeque::with_capacity(max_size),
            max_size,
        }
    }

    fn add(&mut self, experience: Experience) {
        self.memory.push_front(experience);
        self.memory.truncate(self.max_size);
    }

    fn sample(&self, batch_size: usize) -> Vec<Experience> {
        let mut rng = rand::rng();

        self.memory
            .iter()
            .choose_multiple(&mut rng, batch_size)
            .into_iter()
            .cloned()
            .collect()
    }

    fn len(&self) -> usize {
        self.memory.len()
    }
}

#[derive(Debug, Clone)]
struct Experience {
    state: game::StateArray,
    action: usize,
    reward: f32,
    next_state: game::StateArray,
    terminated: bool,
}

pub fn train<B: AutodiffBackend>(device: B::Device) {
    let mut emulator = game::Minesweeper::new_with_mines(constants::MINES);
    let mut config = TrainingConfig::new();
    let mut memory = Memory::new(config.replay_mem_capacity);

    let mut policy_model = ModelConfig::new(
        constants::ROWS * constants::COLS * 10,
        512,
        constants::ROWS * constants::COLS,
    )
    .init::<B>(&device);
    let mut target_model = policy_model.clone();

    let mut rewards_per_episode = vec![0.; config.num_episodes];
    // let mut epsilon_per_episode = Vec::new();

    // B::seed(config.seed);

    let mut step_count = 0;
    for episode in 0..config.num_episodes {
        emulator = game::Minesweeper::new_with_mines(constants::MINES);

        loop {
            // Fill memory with some states
            let possible_actions = emulator
                .opened
                .iter()
                .flatten()
                .enumerate()
                .filter_map(|(i, &open)| if open { None } else { Some(i) })
                .collect::<Vec<_>>();

            assert!(!possible_actions.is_empty());

            let action = if rand::random::<f32>() < config.epsilon {
                possible_actions[rand::random_range(0..possible_actions.len())] // TODO: Can use
            } else {
                let forward = policy_model
                    .clone()
                    .no_grad() // TODO: Enusre this is not persistant
                    .forward(Tensor::from_data(emulator.get_category_vec(), &device));

                let opened_float_map: [f32; constants::ROWS * constants::COLS] = emulator
                    .opened
                    .iter()
                    .flatten()
                    .map(|&a| if a { 0. } else { 1. })
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap();

                let possibility_mask = Tensor::from_data(opened_float_map, &device);

                let masked_forward = forward * possibility_mask;

                masked_forward
                    .argmax(0)
                    .into_data()
                    .to_vec::<i32>()
                    .unwrap()[0] as usize
            };

            // let policy_out = policy_model.forward(emulator.get_category_vec());
            // let target_out = target_model.forward(emulator.get_category_vec());

            let state = emulator.get_category_vec();
            let move_ = emulator.click(action / constants::COLS, action % constants::COLS);

            let reward = get_reward(move_, &emulator);
            rewards_per_episode[episode] += reward;

            let terminated = emulator.is_board_completed() || matches!(move_, game::Square::Mine);
            let experience = Experience {
                state,
                action,
                reward,
                next_state: emulator.get_category_vec(),
                terminated,
            };

            memory.add(experience);

            step_count += 1;

            if terminated {
                break;
            }
        }

        let average_timefrage = 20;
        if episode > average_timefrage {
            let average_reward: f32 = rewards_per_episode
                .iter()
                .skip(episode - average_timefrage)
                .sum::<f32>()
                / average_timefrage as f32;
            println!(
                "[Episode {}] Average Reward {:.3} Epsilon {:.3}",
                episode, average_reward, config.epsilon
            );
        }

        // TODO: Reward tacking

        if memory.len() > config.batch_size {
            let batch = memory.sample(config.batch_size);
            // ==== OPTIMIZING STEPS ====
            let config_optimizer = AdamConfig::new();
            let mut optim = config_optimizer.init();

            let discount_factor_g: f32 = 0.9;
            let mut current_q_list = Vec::new();
            let mut target_q_list: Vec<Tensor<B, 1>> = Vec::new();
            for ex in batch {
                let state_tensor = Tensor::from_data(ex.state, &device);

                let target = if ex.terminated {
                    ex.reward
                } else {
                    let next_q: f32 = target_model
                        .clone() // TODO: oof
                        .no_grad()
                        .forward(Tensor::from_data(ex.next_state, &device))
                        .max()
                        .into_scalar()
                        .elem();
                    ex.reward + discount_factor_g * next_q
                };

                let current_q = policy_model.forward(state_tensor.clone());
                current_q_list.push(current_q);

                let target_q = target_model.forward(state_tensor);
                let target_q = target_q.slice_assign(
                    [ex.action..(ex.action + 1)],
                    Tensor::from_data([target], &device),
                );

                // let mut target_data = target_q.into_data().to_vec::<f32>().unwrap();
                // target_data[ex.action] = target;
                //
                // let target_data: [f32; constants::ROWS * constants::COLS] =
                //     target_data.try_into().unwrap();

                target_q_list.push(target_q);
            }
            let current_q_tensor: Tensor<B, 2> = Tensor::stack(current_q_list, 0);
            let target_q_tensor: Tensor<B, 2> = Tensor::stack(target_q_list, 0);

            let loss =
                MseLoss::new().forward(current_q_tensor, target_q_tensor, nn::loss::Reduction::Sum);
            println!(
                "[Train - Episode {} ] Loss {:.3}",
                episode,
                loss.clone().into_scalar(),
            );

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &policy_model);

            policy_model = optim.step(config.lr, policy_model, grads);

            // END Optimization

            config.epsilon = 0f32.max(config.epsilon - 1. / config.num_episodes as f32);

            if step_count > config.sync_rate {
                target_model = policy_model.clone();
            }
        }

        if episode % config.sync_rate == 0 {
            target_model = policy_model.clone();
        }
    }

    policy_model
        .valid()
        .save_file(format!("model.pt"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");

    plot_rewards_per_episode(rewards_per_episode).unwrap();
}

fn plot_rewards_per_episode(
    rewards_per_episode: Vec<f32>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Smooting of rewards over 10 episodes
    let smoothing_radius = 10;
    let smoothed_rewards: Vec<f32> = rewards_per_episode
        .windows(smoothing_radius)
        .map(|window| window.iter().sum::<f32>() / smoothing_radius as f32)
        .collect();

    const OUT_FILE_NAME: &str = "rewards_per_episode.png";
    let root = BitMapBackend::new(OUT_FILE_NAME, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_reward = smoothed_rewards.iter().cloned().fold(0., f32::max);
    let min_reward = smoothed_rewards.iter().cloned().fold(0., f32::min);
    let mut chart = ChartBuilder::on(&root)
        .caption("Rewards per Episode", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0..smoothed_rewards.len(), min_reward..max_reward)?;

    chart.configure_mesh().draw()?;

    chart.draw_series(LineSeries::new(
        smoothed_rewards
            .iter()
            .enumerate()
            .map(|(i, &reward)| (i, reward)),
        &RED,
    ))?;
    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;

    Ok(())
}

fn get_reward(prev_move: game::Square, emulator: &game::Minesweeper) -> f32 {
    if matches!(prev_move, game::Square::Mine) {
        return -20.;
    }
    if emulator.is_board_completed() {
        return 10.;
    }

    1.
}

fn main() {
    let device = dbg!(Default::default());
    train::<MyAutodiffBackend>(device);
    println!("Done")
}
