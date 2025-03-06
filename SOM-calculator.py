import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import re

class SomGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Self-Organizing Map (SOM) Application")
        self.root.geometry("900x700")
        self.root.configure(bg="#f0f0f0")

        # Main frame
        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Input frame
        input_frame = ttk.LabelFrame(main_frame, text="Input Data")
        input_frame.pack(fill=tk.BOTH, padx=10, pady=10)

        # Training data
        ttk.Label(input_frame, text="Training Data (Enter each vector on a new line, values separated by commas):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.training_data_text = scrolledtext.ScrolledText(input_frame, wrap=tk.WORD, width=40, height=5)
        self.training_data_text.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W+tk.E)
        self.training_data_text.insert(tk.END, "0.8, 0.7, 0.4\n0.6, 0.9, 0.9\n0.3, 0.4, 0.1\n0.1, 0.1, 0.2")

        # Initial weights
        ttk.Label(input_frame, text="Initial Weights (Enter each row on a new line, values separated by commas):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.weights_text = scrolledtext.ScrolledText(input_frame, wrap=tk.WORD, width=40, height=3)
        self.weights_text.grid(row=3, column=0, padx=5, pady=5, sticky=tk.W+tk.E)
        self.weights_text.insert(tk.END, "0.5, 0.6, 0.8\n0.4, 0.2, 0.5")

        # Parameters frame
        params_frame = ttk.Frame(input_frame)
        params_frame.grid(row=4, column=0, sticky=tk.W+tk.E, padx=5, pady=5)

        # Learning rate and iterations input
        ttk.Label(params_frame, text="Learning Rate:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.learning_rate_var = tk.StringVar(value="0.5")
        ttk.Entry(params_frame, textvariable=self.learning_rate_var, width=10).grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(params_frame, text="Iterations:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        self.iterations_var = tk.StringVar(value="1")
        ttk.Entry(params_frame, textvariable=self.iterations_var, width=10).grid(row=0, column=3, padx=5, pady=5)

        # Run button
        self.run_button = ttk.Button(input_frame, text="Run Algorithm", command=self.run_som)
        self.run_button.grid(row=5, column=0, padx=5, pady=10, sticky=tk.E)

        # Results frame
        self.results_frame = ttk.LabelFrame(main_frame, text="Results")
        self.results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Output text
        self.output_text = scrolledtext.ScrolledText(self.results_frame, wrap=tk.WORD, width=50, height=10)
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Visualization frame
        self.viz_frame = ttk.LabelFrame(main_frame, text="Visualization")
        self.viz_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Figure for visualization
        self.fig, self.ax = plt.subplots(figsize=(7, 5))  # تكبير حجم المخطط
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def parse_matrix_input(self, text):
        """Parse the text input into a numpy array"""
        lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
        matrix = []

        for line in lines:
            values = [float(val.strip()) for val in re.split(r',|\s+', line) if val.strip()]
            matrix.append(values)

        return np.array(matrix)

    def find_bmu(self, sample, weights):
        """Find the Best Matching Unit (BMU) for input data"""
        distances = np.linalg.norm(weights - sample, axis=1)
        return np.argmin(distances)

    def run_som(self):
        """Run the SOM algorithm with user input"""
        try:
            training_data = self.parse_matrix_input(self.training_data_text.get("1.0", tk.END))
            weights = self.parse_matrix_input(self.weights_text.get("1.0", tk.END))
            learning_rate = float(self.learning_rate_var.get())
            iterations = int(self.iterations_var.get())

            if len(training_data) == 0 or len(weights) == 0:
                messagebox.showerror("Error", "Please enter training data and weights")
                return

            if training_data.shape[1] != weights.shape[1]:
                messagebox.showerror("Error", "Training data and weights must have the same number of dimensions")
                return

            self.output_text.delete("1.0", tk.END)
            self.output_text.insert(tk.END, "Weights before training:\n")
            self.output_text.insert(tk.END, str(weights) + "\n\n")

            initial_weights = weights.copy()

            for iteration in range(iterations):
                self.output_text.insert(tk.END, f"Iteration {iteration + 1}:\n")

                for i, sample in enumerate(training_data):
                    bmu_index = self.find_bmu(sample, weights)
                    weights[bmu_index] += learning_rate * (sample - weights[bmu_index])
                    self.output_text.insert(tk.END, f"  Sample {i+1}: BMU = {bmu_index}, Updated Weights = {weights[bmu_index]}\n")

                self.output_text.insert(tk.END, "\n")

            self.output_text.insert(tk.END, "Final Weights after Training:\n")
            self.output_text.insert(tk.END, str(weights))

            self.visualize_som(training_data, initial_weights, weights)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def visualize_som(self, training_data, initial_weights, final_weights):
        self.ax.clear()
        self.ax.scatter(training_data[:, 0], training_data[:, 1], c='blue', marker='o', label='Training Data')
        self.ax.scatter(initial_weights[:, 0], initial_weights[:, 1], c='green', marker='^', label='Initial Weights')
        self.ax.scatter(final_weights[:, 0], final_weights[:, 1], c='red', marker='s', label='Final Weights')

        self.ax.legend(loc='upper right', fontsize=10)
        self.ax.grid(True)
        self.fig.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = SomGUI(root)
    root.mainloop()
