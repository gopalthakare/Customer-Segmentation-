import pandas as pd
import tkinter as tk
from tkinter import filedialog, Toplevel
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting

# Create the main window
root = tk.Tk()
root.title("Customer Segmentation")
root.state('zoomed')  # Open in full-screen mode
root.configure(bg="#A3A380")  # Greyish Green Background

# Title Label
title_label = tk.Label(root, text="CUSTOMER SEGMENTATION", font=("Lato", 32, "bold"),
                       bg="#a3a380", fg="black", padx=20, pady=10)
title_label.pack(pady=20)

# Frame for input controls
frame = tk.Frame(root, bg="#D6CE93", padx=20, pady=20)  # Muted Yellow
frame.pack(pady=20)

# CSV File Selection
label_csv = tk.Label(frame, text="Select CSV File:", font=("Arial", 16), bg="#D6CE93", fg="black")
label_csv.grid(row=0, column=0, sticky="w", padx=5, pady=5)
entry_csv = tk.Entry(frame, width=40, font=("Arial", 16), bg="#EFEBCE", fg="black", relief=tk.FLAT)
entry_csv.grid(row=0, column=1, padx=5, pady=5)
btn_browse = tk.Button(frame, text="Browse", font=("Arial", 14), bg="#EE0000", fg="white",
                        relief=tk.FLAT, command=lambda: browse_file())
btn_browse.grid(row=0, column=2, padx=5, pady=5)

# Features Selection
label_features = tk.Label(frame, text="Select Features for Clustering:", font=("Arial", 16), bg="#D6CE93", fg="black")
label_features.grid(row=1, column=0, sticky="w", padx=5, pady=5)
entry_features = tk.Entry(frame, width=40, font=("Arial", 16), bg="#EFEBCE", fg="black", relief=tk.FLAT)
entry_features.grid(row=1, column=1, padx=5, pady=5)

# Number of Clusters Input
label_clusters = tk.Label(frame, text="Number of Clusters:", font=("Arial", 16), bg="#D6CE93", fg="black")
label_clusters.grid(row=2, column=0, sticky="w", padx=5, pady=5)
entry_clusters = tk.Entry(frame, width=10, font=("Arial", 16), bg="#EFEBCE", fg="black", relief=tk.FLAT)
entry_clusters.grid(row=2, column=1, padx=5, pady=5)

# Segment Button
btn_segment = tk.Button(frame, text="Analyse Clusters", font=("Arial", 16, "bold"),
                        bg="#EE0000", fg="white", padx=10, pady=5, relief=tk.FLAT,
                        command=lambda: segment_clusters())
btn_segment.grid(row=3, column=0, columnspan=3, pady=10)

# Output Text Box
text_output = tk.Text(root, height=20, width=100, font=("Consolas", 12),
                      bg="#D6CE93", fg="black", relief=tk.FLAT, borderwidth=5)
text_output.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

# Function to browse for a file
def browse_file():
    filename = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        entry_csv.delete(0, tk.END)
        entry_csv.insert(0, filename)
        auto_fill_features(filename)

# Function to auto-fill feature selection
def auto_fill_features(csv_path):
    try:
        data = pd.read_csv(csv_path)
        numeric_columns = data.select_dtypes(include=['number']).columns.tolist()
        entry_features.delete(0, tk.END)
        entry_features.insert(0, ",".join(numeric_columns))
    except Exception as e:
        print(f"Error reading CSV: {str(e)}")

# Function to segment customers
def segment_clusters():
    try:
        csv_path = entry_csv.get()
        features = entry_features.get()
        k = entry_clusters.get()

        # Load Data
        data = pd.read_csv(csv_path)
        
        # Convert feature names to a list
        feature_list = [f.strip() for f in features.split(",")]

        # Check if features exist in CSV
        for f in feature_list:
            if f not in data.columns:
                print(f"Feature '{f}' not found in dataset!")
                return

        # Extract Features
        X = data[feature_list]

        # Apply K-Means Clustering
        kmeans = KMeans(n_clusters=int(k), random_state=42, n_init=10)
        data['Cluster'] = kmeans.fit_predict(X)

        # Display Summary in Text Box
        text_output.delete(1.0, tk.END)
        text_output.insert(tk.END, "TEXT SUMMARY\n" + "=" * 50 + "\n\n")  # Added "Text Summary" Heading
        
        for cluster_id in range(int(k)):
            cluster_data = data[data['Cluster'] == cluster_id]

            avg_values = {}
            for f in feature_list:
                if "age" in f.lower():  
                    avg_values[f] = int(cluster_data[f].mean())  # Round Age to whole number
                elif "income" in f.lower():  
                    avg_values[f] = f"${int(cluster_data[f].mean())}"  # Add $ to income
                else:
                    avg_values[f] = round(cluster_data[f].mean(), 2)  # Keep 2 decimals for others

            summary = f"Cluster {cluster_id + 1}:\n" + "\n".join([f"- {f}: {v}" for f, v in avg_values.items()]) + f"\n- Total Customers: {len(cluster_data)}\n\n" + "-" * 50 + "\n\n"
            text_output.insert(tk.END, summary)

        # Plot 2D or 3D Graph
        plot_clusters(data, feature_list, int(k), kmeans.cluster_centers_)

    except Exception as e:
        print(f"Error: {str(e)}")

# Function to plot clusters
def plot_clusters(data, features, k, centroids):
    new_window = Toplevel(root)
    new_window.title("Cluster Visualization")
    new_window.geometry("850x650")  # Larger window
    new_window.configure(bg="#A3A380")

    fig = plt.figure(figsize=(8, 6), dpi=100)  # Larger figure

    if len(features) == 2:
        # 2D Scatter Plot
        ax = fig.add_subplot(111)
        scatter = ax.scatter(data[features[0]], data[features[1]], c=data["Cluster"], cmap="viridis", alpha=0.7, s=50)
        
        # Plot centroids
        ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label="Centroids")

        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])
        ax.set_title(f"2D Customer Segmentation (K={k})")
        ax.legend()

    elif len(features) >= 3:
        # 3D Scatter Plot
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data[features[0]], data[features[1]], data[features[2]], c=data["Cluster"], cmap="viridis", alpha=0.7, s=50)

        # Plot centroids
        ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='red', marker='X', s=200, label="Centroids")

        

        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])
        ax.set_zlabel(features[2])
        ax.set_title(f"3D Customer Segmentation (K={k})")
        ax.legend()

    # Embed the graph in Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=new_window)
    canvas.draw()
    canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)

# Run the Tkinter app
root.mainloop()
