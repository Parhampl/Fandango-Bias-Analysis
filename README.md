
# Fandango Bias Analysis

This project investigates potential biases in Fandango's movie rating system compared to other platforms. It includes datasets, a problem definition, and solutions via Python scripts and Jupyter Notebooks.

---

## Repository Structure

### Problem Definition
- `data/problem_fandango_scrape.csv`: The Fandango dataset with displayed ratings and true ratings.
- `data/problem_all_sites_scores.csv`: Dataset comparing ratings from various review platforms.

### Solutions
- `notebooks/solution_fandango_analysis.ipynb`: A detailed analysis notebook addressing the problem.
- `notebooks/solution_fandango_visualizations.ipynb`: A notebook focusing on creating visualizations for insights.
- `scripts/solution_fandango_analysis.py`: A Python script implementing the analysis programmatically.

### Results
- `results/plots/`: Contains key visualizations generated from the analysis.

---

## How to Use

### Prerequisites
- Python 3.8 or later
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `jupyter`

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Fandango-Bias-Analysis.git
   cd Fandango-Bias-Analysis
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Explore the notebooks or run the script:
   ```bash
   jupyter notebook notebooks/solution_fandango_analysis.ipynb
   python scripts/solution_fandango_analysis.py
   ```

---

## Acknowledgments

This project is inspired by 538's article "[Be Suspicious Of Online Movie Ratings, Especially Fandango’s](http://fivethirtyeight.com/features/fandango-movies-ratings/)".
