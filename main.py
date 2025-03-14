import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
import requests
import base64
from PIL import Image, ImageTk

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("LLAVA Model Test")
        
        # Configure root window to use grid
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # API configuration
        self.api_url = "http://localhost:11434/api/generate"
        self.model = "llava"
        
        # Create the main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Add sizegrip to bottom-right corner
        self.sizegrip = ttk.Sizegrip(self.root)
        self.sizegrip.grid(row=0, column=0, sticky="se")
        
        # Configure main frame grid
        self.main_frame.columnconfigure(0, weight=1)
        for i in range(6):
            self.main_frame.rowconfigure(i, weight=0)
        self.main_frame.rowconfigure(5, weight=1)
        
        # Add image selection button
        self.image_button = ttk.Button(self.main_frame, text="Select Image", command=self.select_image)
        self.image_button.grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        # Image preview label
        self.image_preview = ttk.Label(self.main_frame, text="No image selected")
        self.image_preview.grid(row=1, column=0, sticky=tk.W, pady=(0, 10))
        
        # Add a simple label
        self.label = ttk.Label(self.main_frame, text="Enter your prompt:")
        self.label.grid(row=2, column=0, sticky=tk.W, pady=(0, 5))
        
        # Add text input box
        self.text_input = ttk.Entry(self.main_frame, width=50)
        self.text_input.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        # Bind Enter key to submit
        self.text_input.bind("<Return>", lambda event: self.process_query())
        
        # Add submit button
        self.submit_button = ttk.Button(self.main_frame, text="Submit", command=self.process_query)
        self.submit_button.grid(row=3, column=1, padx=(5, 0), pady=(0, 10), sticky=tk.E)
        
        # Add output display label
        self.output_label = ttk.Label(self.main_frame, text="Response:")
        self.output_label.grid(row=4, column=0, sticky=tk.W, pady=(10, 5))
        
        # Add text output display (scrolled text widget)
        self.text_output = scrolledtext.ScrolledText(self.main_frame, width=60, height=15, wrap=tk.WORD)
        self.text_output.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Variables to store the selected image
        self.image_path = None
        self.text_input.focus_set()

    def select_image(self):
        """Open file dialog to select an image"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.PNG *.JPG *.JPEG *.GIF *.BMP")]
        )
        
        if file_path:
            self.image_path = file_path
            # Load and display preview
            try:
                img = Image.open(file_path)
                img = img.resize((100, 100), Image.LANCZOS)  # Create thumbnail
                self.photo = ImageTk.PhotoImage(img)
                self.image_preview.config(image=self.photo, text="")
                
                # Convert image to base64
                with open(file_path, "rb") as img_file:
                    self.image_data = base64.b64encode(img_file.read()).decode('utf-8')
                    
            except Exception as e:
                self.text_output.insert(tk.END, f"Error loading image: {str(e)}\n")
                self.image_preview.config(text="Error loading image")
                self.image_data = None

    def process_query(self):
        """Process the query with Ollama Llava model"""
        prompt = self.text_input.get()
        if not prompt:
            self.text_output.insert(tk.END, "Please enter a prompt\n")
            return
            
        import threading
        threading.Thread(target=self.send_request, args=(prompt,), daemon=True).start()
    
    def send_request(self, prompt):
        """Send request to Ollama API"""
        try:
            # Prepare the request payload
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
            
            # Add image to payload if available
            if self.image_data:
                payload["images"] = [self.image_data]
            
            # Send the request
            response = requests.post(self.api_url, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                self.root.after(0, self.update_output, result.get("response", "No response"))
            else:
                error_msg = f"Error: {response.status_code} - {response.text}"
                self.root.after(0, self.update_output, error_msg)
                
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self.root.after(0, self.update_output, error_msg)
        
        # Re-enable the submit button
        self.root.after(0, lambda: self.submit_button.config(state="normal"))
    
    def update_output(self, text):
        """Update the output text widget with the response"""
        self.text_output.delete(1.0, tk.END)
        self.text_output.insert(tk.END, text)

def main():
    """Main entry point for the application"""
    # Initialize Tk with specific options
    root_window = tk.Tk()
    
    # Set window title
    root_window.title("LLAVA Model Test")
    
    # Set window geometry
    root_window.geometry("600x600")
    
    # Make window resizable - explicitly set both dimensions
    root_window.resizable(width=True, height=True)
    
    # Set minimum window size
    root_window.minsize(400, 400)
    
    # Force update of window manager hints
    root_window.update_idletasks()
    
    # Create the app
    app = App(root_window)
    
    # Start the main loop
    root_window.mainloop()

if __name__ == "__main__":
    main()
