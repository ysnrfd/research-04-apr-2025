import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import os

class LargeFileFinder:
    def __init__(self, root):
        self.root = root
        self.root.title("Large File Finder")

        # Size selection
        size_label = tk.Label(self.root, text="Select File Size:")
        size_label.pack()

        self.size_entry = tk.Entry(self.root, width=100)
        self.size_entry.pack()

        # Find button
        find_button = tk.Button(self.root, text="Find Large Files", command=self.find_large_files)
        find_button.pack()

        # Results label
        self.result_label = tk.Label(self.root, text="Results will appear here", wraplength=400)
        self.result_label.pack()

    def find_large_files(self):
        directory = filedialog.askdirectory()
        if not directory:
            return

        file_size = self.size_entry.get()
        if not file_size:
            messagebox.showwarning("Warning", "Please enter a file size.")
            return

        try:
            size_in_bytes = float(file_size)
        except ValueError:
            messagebox.showwarning("Warning", "Invalid file size format.")
            return

        self.result_label.config(text="Searching for large files...")

        results = self.get_large_files(directory, size_in_bytes)

        if results:
            self.result_label.config(text="Found large files:\n" + results)
        else:
            self.result_label.config(text="No large files found.")

def get_large_files(self, directory, size):
    large_files = []
    for path in Path(directory).iterdir():
        if path.is_file() and path.stat().st_size > size:
            large_files.append(path.as_posix())
    return "\n".join(large_files)


if __name__ == "__main__":
    root = tk.Tk()
    LargeFileFinder(root)
    root.mainloop()
