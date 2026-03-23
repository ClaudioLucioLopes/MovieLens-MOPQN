<!-- # Fairness-Aware Multi-Objective Recommender System (Pareto-DQN)

Traditional recommender systems optimize almost exclusively for single-objective utility (e.g., maximizing click-through rates or engagement). While effective for short-term retention, this approach inherently traps users in semantic **filter bubbles** and leads to extreme popularity bias (ignoring niche content creators).

This repository implements a state-of-the-art **Multi-Objective Reinforcement Learning (MORL)** framework to solve this. We model the recommendation task as a Multi-Objective Markov Decision Process (MOMDP). Using an Item-Centric Pareto-DQN agent, the system learns to navigate the complex Pareto frontier across three non-aggregable objectives:

1. **User Engagement ($r_{\text{eng}}$):** A shaped continuous proxy for utility.
2. **Information Diversity ($r_{\text{div}}$):** Cosine distance in a dense semantic space.
3. **Provider Fairness ($r_{\text{fair}}$):** A logarithmic exposure penalty to incentivize the long-tail.

---

## 🧠 Core Architectural Innovations

### 1. Semantic Embeddings via NLP
Standard collaborative filtering relies on interaction matrices, which lack context. We map items into a continuous **Semantic Space**:
* We aggregate a movie's title, genres, and user-generated tags into a text document.
* We encode this document using a HuggingFace `SentenceTransformer` (`all-MiniLM-L6-v2`) to generate a dense, 384-dimensional embedding ($v_a$).
* Because vectors are $L_2$-normalized, maximizing the Cosine Distance directly rewards the agent for recommending movies with completely different themes, mathematically breaking the filter bubble.

### 2. User State Maintenance & Preference Drift
The user's continuous state ($s_t$) is defined as the **Semantic Centroid** (mean vector) of their historically liked movies. 
* **Markovian Transition:** User interests are not static. When the agent recommends an item, the user's state drifts toward that item:
  $$s_{t+1} = (1 - \alpha)s_t + \alpha v_a$$
* This dynamic "Preference Drift" forces the RL agent to plan sequentially; exploiting engagement now will narrow the user's centroid, making it difficult to gain diversity rewards in the future.

### 3. Overcoming the Cold Start Problem
Standard Deep Q-Networks maintain an output neuron for every item ID. If a new movie is released, the network breaks. 
* By shifting to semantic embeddings, our network achieves **Zero-Shot Generalization**. A brand-new movie with zero historical ratings can be evaluated immediately by passing its textual metadata through the encoder to get $v_a$.

### 4. Item-Centric Q-Learning & Action Bounding
To solve the curse of dimensionality ($|A| > 60,000$), we decouple candidate generation from RL ranking:
* **Stratified Candidate Pooling:** An Approximate Nearest Neighbors (ANN) index fetches the top items closest to the user's state. We explicitly inject a fixed ratio (e.g., 30%) of zero-exposure long-tail items into this pool. This guarantees the agent always has options to optimize $r_{\text{fair}}$.
* **Item-Centric Networks:** Instead of discrete action outputs, the networks take the user state $s_t$ and item embedding $v_a$ as inputs: $f(s_t, v_a) \rightarrow \mathbb{R}^3$. The agent evaluates exactly 100 dynamic candidates per step in a single batched forward pass.

---

## 📂 File Architecture

* **`MovieLensDataHandler.py`**: The data pipeline. Processes CSVs, runs the Sentence Transformer, builds the Semantic KNN index, computes user centroids, and executes Fairness-Aware Candidate Generation.
* **`MovieLensMOEnv.py`**: The Gym-compatible MOMDP environment. Manages sequential interactions, applies preference drift, tracks global fairness exposure, and dynamically injects the `candidate_embeddings` tensor into the observation space.
* **`RewardApproximator.py`**: An Item-Centric neural network that predicts the immediate 3D reward vector by processing the concatenated user state and item embedding.
* **`NonDominatedApproximator.py`**: Estimates the Pareto frontier surface. It takes the user state, item embedding, and sampled objective coordinates to predict non-dominated future returns.
* **`Estimator.py`**: The optimization wrapper that cleanly abstracts PyTorch backpropagation, MSE loss calculation, and target network updates away from the forward-pass architectures.
* **`Pareto.py`**: The MORL Agent controller. Computes the Minkowski sum of rewards and future non-dominated sets. Uses `pymoo` to execute an $\epsilon$-greedy policy by maximizing the **Hypervolume** over the dynamic candidate pool.
* **`StandardDQN.py`**: The baseline agent. Uses an identical Item-Centric architecture but optimizes a scalar Q-value (strictly User Engagement) to serve as our empirical control.
* **`ReplayMemory.py`**: An advanced Experience Replay buffer adapted for continuous action spaces. Stores the chosen item embedding and the full candidate tensor available at the next state to facilitate target evaluations.
* **`main.py`**: The experiment orchestrator. Executes Phase 2 (Training) and Phase 3 (Evaluation), computes covariance matrices, and plots empirical visualizations.

---

## 📊 Experiments & Expected Results

The pipeline (`main.py`) trains both the Pareto-DQN and a Standard DQN baseline, followed by strict deterministic evaluation ($\epsilon = 0.0$). 

### 1. Mitigating the Filter Bubble
* **Metric:** Semantic Variance (Trace of the state Covariance Matrix).
* **Dynamics:** The Standard DQN greedily recommends items mathematically identical to the user's state. Preference drift causes the user's centroid to collapse into a dense micro-cluster (variance approaches 0), empirically proving the filter bubble. The Pareto-DQN actively maximizes $r_{\text{div}}$, maintaining high, stable variance.

### 2. The Price of Responsibility
* **Metric:** 2D and 3D Pareto Frontier Projections.
* **Dynamics:** The Standard DQN clusters at the extreme Engagement axis but scores abysmally on Diversity and Fairness. The Pareto-DQN discovers the Pareto surface. The distance between these clusters accurately quantifies the **Price of Responsibility**—the minimal numerical drop in pure utility required to achieve massive gains in societal value alignment. -->

# Fairness-Aware Multi-Objective Recommender System (Pareto-DQN)

Traditional recommender systems optimize almost exclusively for single-objective utility (e.g., maximizing click-through rates or engagement). While effective for short-term retention, this approach inherently traps users in semantic **filter bubbles** and leads to extreme popularity bias, ignoring niche content creators.

This repository implements a state-of-the-art **Multi-Objective Reinforcement Learning (MORL)** framework. We model the recommendation task as a Multi-Objective Markov Decision Process (MOMDP). Using an Item-Centric Pareto-DQN agent, the system learns to navigate the complex Pareto frontier across three non-aggregable objectives: User Engagement, Information Diversity, and Provider Fairness.

---
## 🧮 Mathematical Fundamentals

The recommendation environment is formalized as a Multi-Objective Markov Decision Process (MOMDP) defined by the tuple $\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathbf{R}, \gamma \rangle$.

### 1. State Space ($\mathcal{S}$)
The user's state $s_t \in \mathbb{R}^d$ is a continuous semantic centroid representing their historical preferences in a dense latent space (where $d = 384$). It is computed by averaging the normalized embeddings of the movies they have liked:

$$s_t = \frac{\sum_{i \in H_t} v_i}{|H_t|}$$

**Where:**
* $s_t$: The continuous user state vector at timestep $t$.
* $H_t$: The historical set of items the user has positively interacted with up to time $t$.
* $v_i$: The $L_2$-normalized dense semantic embedding of item $i$ (extracted via NLP).
* $|H_t|$: The total number of items in the user's history set.

### 2. Action Space ($\mathcal{A}$)
Due to the intractable size of the full item catalog ($|\mathcal{I}| > 60,000$), the action space is dynamically bounded. At each timestep $t$, the environment generates a candidate set $\mathcal{C}_t \subset \mathcal{I}$. 

The agent selects an action $a_t \in \{0, 1, \dots, K-1\}$, which points to a specific item embedding in the candidate set: $v_{a_t} \in \mathcal{C}_t$.

**Where:**
* $\mathcal{I}$: The global catalog of all available items (movies).
* $\mathcal{C}_t$: The dynamically generated subset of candidates at time $t$.
* $K$: The maximum size of the candidate pool (e.g., $100$).
* $v_{a_t}$: The continuous semantic embedding of the specific item chosen by the agent.

### 3. Transition Dynamics ($\mathcal{P}$)
To maintain the Markov property and model shifting user interests, we apply a **Preference Drift** equation upon a successful recommendation. The user's state mathematically drifts towards the recommended item:

$$s_{t+1} = \frac{(1 - \alpha)s_t + \alpha v_{a_t}}{||(1 - \alpha)s_t + \alpha v_{a_t}||_2}$$

**Where:**
* $s_{t+1}$: The new user state for the next timestep.
* $\alpha$: The preference drift rate ($0 \le \alpha \le 1$), controlling how malleable the user's tastes are.
* $s_t$: The current user state.
* $v_{a_t}$: The embedding of the item just recommended to the user.
* $|| \dots ||_2$: The $L_2$-norm operator, ensuring the new state remains on the unit hypersphere so that cosine similarity calculations remain valid.

### 4. Vectorial Reward Function ($\mathbf{R}$)
The environment returns a 3-dimensional reward vector $\mathbf{r}_t = [r_{eng}, r_{div}, r_{fair}]^T \in \mathbb{R}^3$ evaluating the action across three conflicting objectives:

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

$$Q_{set}(s_t, a_i) \leftarrow \mathbf{r}(s_t, a_i) \oplus \gamma ND_t(s_t, a_i)$$

**Where:**
* $Q_{set}(s_t, a_i)$: The set of all Pareto-optimal 3D return vectors if action $a_i$ is taken.
* $\mathbf{r}(s_t, a_i)$: The immediate 3D reward vector.
* $\oplus$: The Minkowski sum operator (vectorial addition of sets).
* $\gamma$: The discount factor ($0 \le \gamma \le 1$) prioritizing immediate vs. future rewards.
* $ND_t(s_t, a_i)$: The non-dominated set of future returns estimated by the neural network.

    The agent selects the action $a_t$ whose $Q_{set}$ maximizes the **Hypervolume** (the multidimensional integral of the objective space bounded by a reference point).
---

## ⚙️ Key Parameters Explained

To successfully tune this system, you must understand the core parameters dictating the agent's behavior:

### 1. Preference Drift Rate (α)
* **Technical Role:** A scalar between 0 and 1 controlling the magnitude of the continuous state transition vector in the environment step.
* **Non-Technical Meaning:** **User Malleability.** How quickly does a user's taste change after watching a recommendation? If this is high, watching a single Sci-Fi movie makes the system think the user *only* wants Sci-Fi now. If it's low, the user has stubborn, established tastes.

### 2. Top-K Candidates (K)
* **Technical Role:** Bounding the discrete action space to ensure computational tractability during the batched neural network forward pass `f(s_t, v_a)`.
* **Non-Technical Meaning:** **The Consideration Pool.** Instead of evaluating all 60,000 movies at once (which would freeze the server), the AI narrows its focus down to a "Top 100" list at any given second, picking the absolute best option from that curated list.

### 3. Fairness Ratio (ρ)
* **Technical Role:** Dictates the stratified sampling constraint during candidate generation. This guarantees the action space contains items capable of minimizing the Gini Coefficient.
* **Non-Technical Meaning:** **The "Hidden Gem" Quota.** If set to 30%, it forces the system to include at least 30 movies from lesser-known, niche creators in the AI's consideration pool, ensuring unpopular movies always get a fighting chance to be recommended.

### 4. Discount Factor (γ)
* **Technical Role:** Standard RL parameter controlling the weighting of the Non-Dominated future returns in the Minkowski sum.
* **Non-Technical Meaning:** **The AI's Patience.** A high value means the AI is willing to sacrifice a quick, guaranteed click right now if it calculates that taking a slight risk will lead to a healthier, more diverse, and fairer user journey in the long term.

---

## 🧠 Core Architectural Innovations

### 1. Semantic Embeddings via NLP
Standard collaborative filtering relies on interaction matrices, which lack context. We map items into a continuous semantic space using a HuggingFace `SentenceTransformer` (`all-MiniLM-L6-v2`) to generate dense, 384-dimensional embeddings ($v_a$) from movie metadata.

### 2. Overcoming the Cold Start Problem
Standard Deep Q-Networks maintain an output neuron for every item ID. If a new movie is released, the network breaks. By shifting to semantic embeddings, our network achieves **Zero-Shot Generalization**. A brand-new movie with zero historical ratings can be evaluated immediately by passing its textual metadata through the encoder.

### 3. Item-Centric Q-Learning & Fairness-Aware Candidate Bounding
To solve the curse of dimensionality, we decouple candidate generation from RL ranking:
* **Stratified Candidate Pooling:** An Approximate Nearest Neighbors (ANN) index fetches items closest to the user's state. We explicitly inject a fixed ratio (e.g., 30%) of zero-exposure long-tail items into this pool, guaranteeing the agent has pathways to optimize $r_{fair}$.
* **Item-Centric Networks:** Instead of discrete action outputs, the networks take the user state $s_t$ and item embedding $v_a$ as inputs: $f(s_t, v_a) \rightarrow \mathbb{R}^3$. The agent evaluates exactly 100 dynamic candidates per step in a single batched forward pass.
---

## 📂 File Architecture

* **`MovieLensDataHandler.py`**: The data pipeline. Processes CSVs, runs the Sentence Transformer, builds the Semantic KNN index, and executes Fairness-Aware Candidate Generation.
* **`MovieLensMOEnv.py`**: The Gym-compatible MOMDP environment. Manages sequential interactions, applies preference drift, tracks global fairness exposure, and dynamically injects embeddings into the observation space.
* **`RewardApproximator.py`**: An Item-Centric PyTorch network that predicts the immediate 3D reward vector by processing the concatenated user state and item embedding.
* **`NonDominatedApproximator.py`**: Estimates the Pareto frontier surface to predict non-dominated future returns.
* **`Estimator.py`**: The optimization wrapper abstracting MSE loss calculation and target network updates away from the forward-pass architectures.
* **`Pareto.py`**: The MORL Agent controller. Uses `pymoo` to execute an ε-greedy policy by maximizing the Hypervolume over the dynamic candidate pool.
* **`StandardDQN.py`**: The baseline agent. Uses an identical Item-Centric architecture but optimizes a scalar Q-value (strictly User Engagement) as an empirical control.
* **`ReplayMemory.py`**: Experience Replay buffer adapted for continuous action spaces.
* **`main.py`**: The experiment orchestrator. Executes Training and Evaluation, computes covariance matrices, and plots empirical visualizations.

---

## 📊 Experiments & Expected Results

The pipeline trains both the Pareto-DQN and a Standard DQN baseline, followed by strict deterministic evaluation (`ε = 0`). 

### 1. Mitigating the Filter Bubble
* **Metric:** Semantic Variance (Trace of the state Covariance Matrix).
* **Dynamics:** The Standard DQN greedily recommends items mathematically identical to the user's state. Preference drift causes the user's centroid to collapse into a dense micro-cluster. The Pareto-DQN actively maximizes diversity, maintaining high, stable variance.

### 2. The Price of Responsibility
* **Metric:** 2D and 3D Pareto Frontier Projections.
* **Dynamics:** The Standard DQN clusters at the extreme Engagement axis but scores abysmally on Diversity and Fairness. The Pareto-DQN discovers the Pareto surface. The distance between these clusters accurately quantifies the **Price of Responsibility**—the minimal numerical drop in pure utility required to achieve massive gains in societal value alignment.