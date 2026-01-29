# Celeste Golden Modeler

Have you ever wondered whether you were grinding for golden strawberries the right way? That your efforts are misplaced and in need of a little mathematical optimization? Well, wonder no more. As with most things, the computer is probably better at grindstrats than you are.

![Overall mean time to completion](https://i.imgur.com/bmw8ufx.png)
![Individual room modeling](https://i.imgur.com/7EhEWes.png)
![Histogram of completion times](https://i.imgur.com/ISSVDoB.png)

## Overview

This project fits logistic regression models to gameplay attempt data from Celeste's golden strawberry challenges. It uses these models to simulate and compare different practice strategies, determining which approaches minimize expected completion time.

**Basically:** as players practice difficult rooms, their probability of success increases over time. By modeling this learning curve for each room, we can evaluate whether targeted practice of weak rooms or other strategies are more efficient than naive grinding.

## Features

- **Logistic model fitting**: Fits learning curves ([logistic sigmoids](https://en.wikipedia.org/wiki/Logistic_function)) to attempt success/failure data using [maximum likelihood estimation](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation)
- **Visualization**: Generates plots showing learning progression for individual rooms and comparative analysis across all rooms
- **Strategy simulation**: Monte Carlo simulation of different practice approaches
- **Strategy comparison**: Statistical comparison of expected completion times across strategies

## Installation

Requires Python 3.8+ with the following dependencies:

```bash
pip install numpy scipy matplotlib
```

## Usage

### Basic Usage

Run both analysis and benchmarking with default settings:

```bash
python main.py
```

### Analysis Only

Fit models and generate learning curve visualizations:

```bash
python main.py --analyze
```

### Benchmarking Only

Run strategy simulations (requires pre-computed models):

```bash
python main.py --benchmark
```

### Advanced Options

```bash
python main.py --attempts inputs/attempts.json \
               --times inputs/times.json \
               --simulations 5000 \
               --chunks 1 5 7 10
```

- `--attempts`: Path to attempts JSON file
- `--times`: Path to room completion times JSON file
- `--simulations`: Number of Monte Carlo simulations (default: 1000)
- `--chunks`: Chunk sizes for backward learning variants (default: 7)

## Input Format

### attempts.json

Maps room IDs to lists of boolean success/failure outcomes:

```json
{
  "00": [false, false, true, true, false, ...],
  "01": [false, true, false, ...]
}
```

For each room, this is the sequence of successes/deaths one had across the entire golden grind session, according to an observer who could only see your gameplay in that specific room.

### times.json

Maps room IDs to completion times in seconds:

```json
{
  "00": 8.5,
  "01": 13.5
}
```

For each room, this is the amount of time a successful run-through of that room is expected to take. (See *Assumptions and Limitations* below.)

## Strategies

The framework implements and compares the following practice strategies:

1. **Naive Grind**: Attempt full runs from start; reset to first room on any death
2. **Cyclic Grind**: Practice rooms in order, cycling continuously without resets
3. **Backward Learning**: Master final rooms first, gradually prepend earlier rooms
4. **Semiomniscient**: Uses fitted models to dynamically choose between targeted practice and full run attempts based on expected benefit

## Output

### Analysis Outputs

- `plots/analysis/room_*.png`: Individual room learning curves
- `plots/analysis/summary_all_rooms.png`: Comparative visualization of all rooms
- `data/model_parameters.json`: Fitted model parameters for all rooms

### Benchmark Outputs

- `plots/benchmark/time_distribution.png`: Histogram of completion times by strategy
- `plots/benchmark/mean_time_comparison.png`: Bar chart comparing expected times
- `plots/benchmark/room_*_attempts.png`: Per-room attempt counts by strategy
- `data/benchmark_results.json`: Detailed statistical results

## Code Structure

- `main.py`: Command-line interface and workflow orchestration
- `models.py`: Logistic regression model fitting and probability calculations
- `strategies.py`: Practice strategy implementations
- `simulator.py`: Monte Carlo simulation engine
- `analysis.py`: Model fitting and visualization generation
- `benchmark.py`: Strategy comparison and benchmark plotting

## Model Details

Success probability follows a logistic model:

```
P(success | attempt n) = 1 / (1 + exp(-(β₀ + β₁ · n)))
```

Where:
- β₀: Initial skill level (log-odds of success on first attempt)
- β₁: Learning rate (change in log-odds per attempt)

Models are fit via maximum likelihood estimation. If a negative learning rate is detected, the model falls back to a constant probability fit in order to avoid non-terminating simulations.

## Assumptions and Limitations

The math relies on several simplifying assumptions that may not fully capture the complexity of skill acquisition in practice:

**Learning Model Assumptions**:
- Success probability follows a monotonic logistic curve over attempts. In reality, players may experience plateaus, temporary regression, fatigue effects, or non-linear breakthroughs when they finally take a single second to realize they want a diagonal dash instead of an updash.
- It is said you learn more when you fail than when you succeed. Regardless, a successful attempt sees equal progression along the learning curve as a failed attempt.
- Each attempt is independent given the attempt number. The model does not account for tilt, warm-up periods, session-to-session variance, or ultra instinct gamer lock-in where you defy statistics to manifest your own escape from this golden berry hell.

**Strategy Evaluation Assumptions**:
- The "semiomniscient" strategy assumes perfect knowledge of model parameters, which would not be available to a real player, or at least not ahead of time. Investigation of an [online learning approach](https://en.wikipedia.org/wiki/Online_machine_learning) would be of interest.
- Switching from level to level is assumed to be cheap and easy, e.g. zero-cost usage of the debug map in Everest.
- To get the golden berry is *defined as* completing every room in order without dying. That is, one does need to commit to the golden berry ahead of an attempt.

**Practical Limitations**:
- A successful attempt at a room is assumed to always take the same amount of time. In reality, better strats and execution lead to faster times.
- The model assumes failed attempts always end at the halfway point (death time = 0.5 × completion time).
- Mental factors are not modeled, such as performance anxiety, or the psychological and nutritional impact of long grinding sessions.
- The framework does not account for meta-strategic decisions like when to take breaks or how loud to scream at 2am when a pink cloud eats your jump.

## License

This project is provided as-is for analysis and educational purposes. I stake no claim and deem this free to use or modify however, without attribution.