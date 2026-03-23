# Pareto-DQN for Fairness-Aware Multi-Objective Recommender Systems

This repository implements a state-of-the-art Multi-Objective Reinforcement Learning (MORL) framework designed to mitigate filter bubbles and promote provider fairness in recommender systems. 

Unlike traditional single-objective Deep Q-Networks (DQN) that over-optimize for user engagement (click-through rates), this framework models the recommendation task as a Multi-Objective Markov Decision Process (MOMDP). The agent learns to approximate the Pareto frontier across three conflicting objectives: **Engagement** ($r_{\text{eng}}$), **Information Diversity** ($r_{\text{div}}$), and **Provider Fairness** ($r_{\text{fair}}$).

To handle the massive discrete action space of the MovieLens catalog, the architecture utilizes **Semantic Candidate Generation** and an **Item-Centric Q-Network**.

## 📂 File Structure & Roles

### 1. Data Pipeline & Candidate Generation
* **`MovieLensDataHandler.py`**: The backbone of the data pipeline. 
    * **Role**: Processes raw `movies.csv`, `tags.csv`, and `ratings.csv`. It utilizes a HuggingFace `SentenceTransformer` to project items into a dense semantic space, generating embeddings ($\mathbf{v}_a$). It computes continuous user states ($\mathbf{s}_t$) by calculating the semantic centroid of a user's positive interactions.
    * **Candidate Generation**: Solves the $|A|$ explosion problem using a K-Nearest Neighbors (KNN) index. It implements *Stratified Candidate Pooling* to inject long-tail items into the candidate pool, ensuring the agent always has a pathway to optimize the Provider Fairness objective.

### 2. The RL Environment
* **`MovieLensMOEnv.py`**: The OpenAI Gym / Gymnasium compatible MOMDP environment.
    * **Role**: Manages the sequential interaction between the user and the agent. 
    * **State & Action**: The state is the continuous user centroid. The action is a dynamically mapped index corresponding to the Top-K generated candidates.
    * **Dynamics**: Simulates **Preference Drift**—updating the user's semantic centroid towards accepted recommendations to maintain the Markov property.
    * **Reward**: Computes the vectorized reward $\mathbf{r}_t \in \mathbb{R}^3$:
        * Utility: Shaped continuous proxy using the dot product sigmoid.
        * Diversity: Cosine distance $1 - \cos(\mathbf{s}_t, \mathbf{v}_a)$.
        * Fairness: Logarithmic exposure penalty $\frac{1}{\log(1 + \text{Count}(a_t))}$.

### 3. The Item-Centric Neural Networks
* **`RewardApproximator.py`**: 
    * **Role**: A continuous regression network predicting the immediate 3D reward vector $\mathbf{\bar{r}}(s_t, v_a)$. Instead of using one-hot action indices, it concatenates the user state and the specific item embedding to evaluate the latent collaborative/semantic match.
* **`NonDominatedApproximator.py`**:
    * **Role**: A network estimating the $d$-th dimension of the Non-Dominated future returns. By taking the user state, item embedding, and $d-1$ sampled objective coordinates, it maps out the continuous Pareto frontier surface for any state-action pair.

### 4. Agent & Optimization Logic
* **`Estimator.py`**: 
    * **Role**: The optimization wrapper for the PyTorch networks. Following Separation of Concerns (SoC), it isolates the Mean Squared Error (MSE) loss calculations, Adam optimizer steps, and Polyak/Hard target network updates from the forward-pass architectures.
* **`Pareto.py`**: 
    * **Role**: The Pareto-DQN Agent controller. It receives the batch of continuous candidate embeddings from the environment and passes them through the `Estimator`s. It computes the Non-Dominated sorting via objective sampling and calculates the Hypervolume indicator for action selection using an $\epsilon$-greedy policy.

---

## 🔄 System Interaction (The Data Flow)

The execution flow per timestep $t \to t+1$ operates as follows:

1.  **State Observation**: The environment (`MovieLensMOEnv.py`) outputs the current user centroid $\mathbf{s}_t$ and an `info` dictionary containing a dynamic tensor of 100 candidate item embeddings ($\mathbf{v}_a$).
2.  **Pareto Evaluation**: The Agent (`Pareto.py`) passes $\mathbf{s}_t$ and the candidate batch through `RewardApproximator` and `NonDominatedApproximator`. It generates a set of future return vectors ($Q_{set}$) for each of the 100 candidates.
3.  **Hypervolume Selection**: The Agent computes the multi-dimensional volume (Hypervolume) of each candidate's $Q_{set}$ against a reference point. It selects the candidate index that maximizes this volume (with $\epsilon$ exploration).
4.  **Transition**: The index is passed back to the environment. The environment logs the exposure (updating fairness counts), calculates the 3D reward $\mathbf{r}_t$, applies preference drift to $\mathbf{s}_t$, and asks the `MovieLensDataHandler` to generate a fresh batch of 100 candidates for $t+1$.