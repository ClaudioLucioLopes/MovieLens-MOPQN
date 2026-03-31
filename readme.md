# Fairness-Aware Multi-Objective Recommender System (Pareto-DQN)

  
Traditional recommender systems optimize almost exclusively for single-objective utility (e.g., maximizing click-through rates or engagement). While effective for short-term retention, this approach inherently traps users in semantic **filter bubbles** and leads to extreme popularity bias, ignoring niche content creators. 

This repository implements a state-of-the-art **Multi-Objective Reinforcement Learning (MORL)** framework. We model the recommendation task as a Multi-Objective Markov Decision Process (MOMDP), evaluating three distinct paradigms to navigate the complex Pareto frontier across three non-aggregable objectives: **User Engagement**, **Information Diversity**, and **Provider Fairness**.
  
---

##  Mathematical Fundamentals

  

The recommendation environment is formalized as a Multi-Objective Markov Decision Process (MOMDP) defined by the tuple $\langle  \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathbf{R}, \gamma  \rangle$.

  

### 1. State Space ($\mathcal{S}$)

The user's state $s_t \in  \mathbb{R}^d$ is a continuous semantic centroid representing their historical preferences in a dense latent space (where $d = 384$). It is computed by averaging the normalized embeddings of the movies they have liked:

  

$$s_t = \frac{\sum_{i \in H_t} v_i}{|H_t|}$$

  

**Where:**

* $s_t$: The continuous user state vector at timestep $t$.

* $H_t$: The historical set of items the user has positively interacted with up to time $t$.

* $v_i$: The $L_2$-normalized dense semantic embedding of item $i$ (extracted via NLP).

* $|H_t|$: The total number of items in the user's history set.

  

### 2. Action Space ($\mathcal{A}$)

Due to the intractable size of the full item catalog ($|\mathcal{I}| > 60,000$), the action space is dynamically bounded. At each timestep $t$, the environment generates a candidate set $\mathcal{C}_t \subset  \mathcal{I}$.

  

The agent selects an action $a_t \in \{0, 1, \dots, K-1\}$, which points to a specific item embedding in the candidate set: $v_{a_t} \in  \mathcal{C}_t$.

  

**Where:**

* $\mathcal{I}$: The global catalog of all available items (movies).

* $\mathcal{C}_t$: The dynamically generated subset of candidates at time $t$.

* $K$: The maximum size of the candidate pool (e.g., $100$).

* $v_{a_t}$: The continuous semantic embedding of the specific item chosen by the agent.


### 3. Transition Dynamics & Preference Drift ($\mathcal{P}$) 

To maintain the Markov property and model shifting user interests, we apply a **Preference Drift** equation upon a successful recommendation. The user's state mathematically drifts towards the recommended item:

  

$$s_{t+1} = \frac{(1 - \alpha)s_t + \alpha v_{a_t}}{||(1 - \alpha)s_t + \alpha v_{a_t}||_2}$$

  

**Where:**

* $s_{t+1}$: The new user state for the next timestep.

* $\alpha$: The preference drift rate ($0  \le  \alpha  \le  1$), controlling how malleable the user's tastes are.

* $s_t$: The current user state.

* $v_{a_t}$: The embedding of the item just recommended to the user.

* $|| \dots ||_2$: The $L_2$-norm operator, ensuring the new state remains on the unit hypersphere so that cosine similarity calculations remain valid.

  

### 4. Vectorial Reward Function ($\mathbf{R}$)

The environment returns a 3-dimensional reward vector $\mathbf{r}_t = [r_{eng}, r_{div}, r_{fair}]^T \in  \mathbb{R}^3$ evaluating the action across three conflicting objectives:

  

**I. Engagement (Utility):** A shaped continuous proxy for click-through likelihood.

$$r_{eng}(s_t, v_{a_t}) = \frac{1}{1 + e^{-(s_t \cdot v_{a_t})}}$$

* $s_t \cdot v_{a_t}$: The dot product between the user's state and the item embedding, representing their latent semantic affinity.

  

**II. Information Diversity:** Measures semantic distance to prevent filter bubbles.

$$r_{div}(s_t, v_{a_t}) = 1 - \cos(s_t, v_{a_t})$$

* $\cos(s_t, v_{a_t})$: The cosine similarity between the user's history centroid and the recommended item. Subtracting it from $1$ rewards the agent for recommending orthogonal (unrelated) topics.

  

**III. Provider Fairness:** A logarithmic penalty mitigating popularity bias.

$$r_{fair}(a_t) = \frac{1}{\log(1 + C(a_t))}$$

* $C(a_t)$: The global historical exposure count of the item $a_t$ across all users in the environment. Highly recommended blockbusters yield near-zero fairness reward, while niche items yield high rewards.

  

### 5. Pareto Optimality and Hypervolume

Because the objectives conflict, the agent calculates the non-dominated set of future returns ($Q_{set}$) using the Minkowski sum:

  

$$Q_{set}(s_t, a_i) \leftarrow  \mathbf{r}(s_t, a_i) \oplus  \gamma ND_t(s_t, a_i)$$

  

**Where:**

* $Q_{set}(s_t, a_i)$: The set of all Pareto-optimal 3D return vectors if action $a_i$ is taken.

* $\mathbf{r}(s_t, a_i)$: The immediate 3D reward vector.

* $\oplus$: The Minkowski sum operator (vectorial addition of sets).

* $\gamma$: The discount factor ($0  \le  \gamma  \le  1$) prioritizing immediate vs. future rewards.

* $ND_t(s_t, a_i)$: The non-dominated set of future returns estimated by the neural network.
  
The Pareto DQN agent selects the action $a_t$ whose $Q_{set}$ maximizes the **Hypervolume** (the multidimensional integral of the objective space bounded by a reference point).



### 6. Multi-Objective Actor-Critic (MOAC) with Envelope RL

The **Envelope MOAC** framework extends the many-objective paradigm by learning a preference-conditioned vectorized value function. This allows a single model to approximate the entire Pareto front by treating the preference vector $w$ as an explicit input.

#### I. Preference-Conditioned Vectorized Critic
Unlike set-based methods, the Critic predicts a 3D vector of expected returns $\mathbf{Q}(s, a, w) \in \mathbb{R}^3$ for a specific state-action pair, conditioned on a preference vector $w$:

$$\mathbf{Q}(s, a, w) = \mathbb{E} \left[ \sum_{t=0}^{\infty} \gamma^t \mathbf{r}_t \mid s_0=s, a_0=a, w \right]$$

**Where:**
* **$w \in \mathbb{R}^3$**: A preference vector sampled from a Dirichlet distribution ($w \sim \text{Dir}(1,1,1)$) during training to ensure uniform coverage of the objective space.
* **$\mathbf{Q}(s, a, w)$**: The predicted returns for Engagement, Diversity, and Fairness simultaneously.

#### II. The Envelope Bellman Equation
The Critic is optimized using the **Envelope Bellman Operator**, which identifies the maximum utility across the learned frontier for the next state s':

$$\mathbf{Q}(s, a, w) \leftarrow \mathbf{r} + \gamma \mathbf{Q}(s', \pi(s', w^{*}), w^{*})$$

$$w^* = \arg\max_{w' \in \mathcal{W}} (w \cdot \mathbf{Q}(s', \pi(s', w'), w'))$$

**Where:**
* **$\mathcal{W}$**: A set of $M$ randomly sampled preference vectors (the "Envelope") used to find the upper bound of the value function.
* **$w^*$**: The preference that maximizes scalarized utility relative to the agent's current objective $w$.

#### III. Stochastic Policy and Entropy Regularization
To prevent policy collapse and maintain high semantic variance, the Actor implements a stochastic policy $\pi_\theta(a|s, w)$[cite: 12]. [cite_start]The Actor maximizes a combination of scalarized utility and policy entropy:

$$J(\theta) = \mathbb{E} [ w \cdot \mathbf{Q}(s, a, w) - \alpha \log \pi_\theta(a|s, w) ]$$.

**Where:**
* **$w \cdot \mathbf{Q}(s, a, w)$**: The scalarized utility (Exploitation).
* **$\alpha \log \pi_\theta(a|s, w)$**: The entropy term scaled by temperature $\alpha$, which rewards the agent for maintaining behavioral diversity (Exploration).

#### IV. Continuous Action Sampling (Reparameterization)
Because the Actor operates in a continuous embedding space, it utilizes the **Reparameterization Trick** to remain differentiable while sampling:

$$a = \tanh(\mu(s, w) + \sigma(s, w) \cdot \epsilon), \quad \epsilon \sim \mathcal{N}(0, I)$$

**Where:**
* **$\mu, \sigma$**: The mean and standard deviation predicted by the Actor network.
* **$\tanh$**: The squashing function ensuring the action vector remains within the valid semantic hypersphere bounds.


___
##  Developed Agents: A Comparative Overview

This repository implements and evaluates three distinct reinforcement learning architectures, progressing from a traditional engagement-centric baseline to a state-of-the-art preference-conditioned framework.

### 1. Standard DQN (Baseline Approach)
The **Standard DQN** represents the conventional approach to recommendation, optimizing strictly for a single objective: **User Engagement**.
* **Architecture**: It utilizes an **Item-Centric Q-Network** that predicts a scalar value for a given user state and item embedding.
* **Learning Goal**: The agent updates its policy using a standard scalar Bellman equation to maximize long-term click-through probability.
* **Limitation**: By over-optimizing for immediate engagement, this agent serves as the empirical control for the formation of **filter bubbles** and semantic homogenization.

### 2. Pareto-DQN (Many-Objective Approach)
The **Pareto-DQN** is a many-objective reinforcement learning agent designed to navigate the complex trade-offs between platform retention and societal values.
* **Vectorial Rewards**: Unlike the baseline, this agent receives a 3D vectorial reward representing Engagement, Diversity, and Fairness.
* **Set-Based Estimation**: It utilizes two primary neural networks—a **Reward Estimator** ($\bar{R}$) and a **Non-Dominated Estimator** ($ND_t$)—to approximate the Pareto frontier surface.
* **Action Selection**: The agent selects actions by calculating the **Minkowski sum** of predicted rewards and future returns, choosing the item that maximizes the **Hypervolume (HV)** indicator.

### 3. Envelope MOAC (Advanced MORL Approach)
The **Envelope Multi-Objective Actor-Critic (MOAC)** is the most advanced agent in this framework, transitioning from discrete set-operations to a scalable, preference-conditioned continuous policy.
* **Preference-Conditioning**: Both the Actor and Critic networks are conditioned on a **preference vector ($w$)**, allowing a single model to approximate the entire Pareto front across any arbitrary priority distribution.
* **Envelope RL**: It utilizes the **Envelope Bellman Equation** to optimize for the convex hull of the objective space, ensuring alignment with the platform's current value priorities.
* **Stochastic Exploration**: The Actor implements a **stochastic policy with entropy regularization** ($\alpha$) to maintain high semantic variance and prevent policy collapse.
* **Reparameterization**: It employs the **reparameterization trick** to remain differentiable while sampling actions from a Gaussian distribution in the 384-dimensional semantic embedding space.


---
## Key Parameters Explained

To successfully tune this system, you must understand the core parameters dictating the behavior of the environment and each specific agent.

### 1. Global Environment Parameters
These parameters are defined in the `MovieLensMOEnv` and affect the underlying simulation dynamics for all agents.

* **Preference Drift Rate ($\alpha$):**
    * **Technical Role:** A scalar between 0 and 1 controlling the magnitude of the continuous state transition vector in the environment step.
    * **Meaning:** **User Malleability.** Determines how quickly a user's semantic centroid shifts toward a recommended item. High values simulate "fast-burn" interest changes, while low values represent stable, long-term preferences.
* **Top-K Candidates ($K$):**
    * **Technical Role:** Bounding the discrete action space to ensure computational tractability during the batched neural network forward pass.
    * **Meaning:** **The Consideration Pool.** The number of items the agent ranks at each step. Increasing $K$ provides more variety but increases inference latency.
* **Fairness Ratio ($\rho$):**
    * **Technical Role:** Dictates the stratified sampling constraint during candidate generation to include long-tail items.
    * **Meaning:** **The "Hidden Gem" Quota.** Ensures that a fixed percentage of the candidate pool consists of niche items with low historical exposure, guaranteeing the agent can always choose to optimize for provider fairness.

### 2. Standard DQN Parameters
Parameters specific to the single-objective engagement baseline.

* **Discount Factor ($\gamma$):**
    * **Technical Role:** Controls the weighting of future engagement rewards. Set to $0.98$ to prioritize long-term session retention.
* **Exploration Schedule ($\epsilon$, $\epsilon_{decay}$):**
    * **Technical Role:** Manages the transition from random item selection to greedy engagement maximization.

### 3. Pareto-DQN Specific Parameters
Parameters governing the set-based many-objective optimization process.

* **Reference Point:**
    * **Technical Role:** The coordinate ($[0,0,0]$) used to calculate the **Hypervolume (HV)**.
    * **Meaning:** Acts as the "floor" for rewards; the agent seeks to maximize the volume of the space dominated by its predicted Pareto set relative to this point.
* **Frontier Samples ($N$):**
    * **Technical Role:** The number of points sampled on the $(d-1)$ dimensional hyper-plane to approximate the non-dominated set.
    * **Meaning:** Controls the resolution of the predicted Pareto front; higher values provide more accurate trade-off estimations at the cost of computation.

### 4. Envelope MOAC Specific Parameters
Parameters for the preference-conditioned stochastic actor-critic.

* **Entropy Coefficient ($\alpha_{entropy}$):**
    * **Technical Role:** Scales the importance of the policy entropy term in the Actor loss function.
    * **Meaning:** **The Randomness Driver.** Prevents the policy from becoming overly deterministic (collapsing). It forces the agent to keep exploring different semantic regions even when a strong preference $w$ is provided.
* **Preference Samples ($M$):**
    * **Technical Role:** The number of random preference vectors sampled during the **Envelope Update** to find the maximum utility frontier.
    * **Meaning:** Determines how thoroughly the agent checks for better alternative alignments in the next state to ensure it stays on the true convex hull of the Pareto front.---

  

## 🧠 Core Architectural Innovations

  

### 1. Semantic Embeddings via NLP

Standard collaborative filtering relies on interaction matrices, which lack context. We map items into a continuous semantic space using a HuggingFace `SentenceTransformer` (`all-MiniLM-L6-v2`) to generate dense, 384-dimensional embeddings ($v_a$) from movie metadata.

  

### 2. Overcoming the Cold Start Problem

Standard Deep Q-Networks maintain an output neuron for every item ID. If a new movie is released, the network breaks. By shifting to semantic embeddings, our network achieves **Zero-Shot Generalization**. A brand-new movie with zero historical ratings can be evaluated immediately by passing its textual metadata through the encoder.

  

### 3. Item-Centric Q-Learning & Fairness-Aware Candidate Bounding

To solve the curse of dimensionality, we decouple candidate generation from RL ranking:

*  **Stratified Candidate Pooling:** An Approximate Nearest Neighbors (ANN) index fetches items closest to the user's state. We explicitly inject a fixed ratio (e.g., 30%) of zero-exposure long-tail items into this pool, guaranteeing the agent has pathways to optimize $r_{fair}$.

*  **Item-Centric Networks:** Instead of discrete action outputs, the networks take the user state $s_t$ and item embedding $v_a$ as inputs: $f(s_t, v_a) \rightarrow  \mathbb{R}^3$. The agent evaluates exactly 100 dynamic candidates per step in a single batched forward pass.

---

  

## 📂 File Architecture

  

*  **`MovieLensDataHandler.py`**: The data pipeline. Processes CSVs, runs the Sentence Transformer, builds the Semantic KNN index, and executes Fairness-Aware Candidate Generation.

*  **`MovieLensMOEnv.py`**: The Gym-compatible MOMDP environment. Manages sequential interactions, applies preference drift, tracks global fairness exposure, and dynamically injects embeddings into the observation space.

*  **`RewardApproximator.py`**: An Item-Centric PyTorch network that predicts the immediate 3D reward vector by processing the concatenated user state and item embedding.

*  **`NonDominatedApproximator.py`**: Estimates the Pareto frontier surface to predict non-dominated future returns.

*  **`Estimator.py`**: The optimization wrapper abstracting MSE loss calculation and target network updates away from the forward-pass architectures.

*  **`Pareto.py`**: The MORL Agent controller. Uses `pymoo` to execute an ε-greedy policy by maximizing the Hypervolume over the dynamic candidate pool.

*  **`StandardDQN.py`**: The baseline agent. Uses an identical Item-Centric architecture but optimizes a scalar Q-value (strictly User Engagement) as an empirical control.

*  **`ReplayMemory.py`**: Experience Replay buffer adapted for continuous action spaces.

*  **`main.py`**: The experiment orchestrator. Executes Training and Evaluation, computes covariance matrices, and plots empirical visualizations.

  

---

  

## 📊 Experiments & Expected Results

  

The pipeline trains both the Pareto-DQN and a Standard DQN baseline, followed by strict deterministic evaluation (`ε = 0`).

  

### 1. Mitigating the Filter Bubble

*  **Metric:** Semantic Variance (Trace of the state Covariance Matrix).

*  **Dynamics:** The Standard DQN greedily recommends items mathematically identical to the user's state. Preference drift causes the user's centroid to collapse into a dense micro-cluster. The Pareto-DQN actively maximizes diversity, maintaining high, stable variance.

  

### 2. The Price of Responsibility

*  **Metric:** 2D and 3D Pareto Frontier Projections.

*  **Dynamics:** The Standard DQN clusters at the extreme Engagement axis but scores abysmally on Diversity and Fairness. The Pareto-DQN discovers the Pareto surface. The distance between these clusters accurately quantifies the **Price of Responsibility**—the minimal numerical drop in pure utility required to achieve massive gains in societal value alignment.
