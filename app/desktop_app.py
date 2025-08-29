"""
Desktop application for offline voice translation
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import queue
import json
import os
import sys
from pathlib import Path
import logging
import webbrowser
from typing import Optional, Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings
from app.core.offline_manager import offline_manager
from app.core.translation_engine import RealTimeTranslationEngine
from app.core.audio_processor import AudioProcessor

logger = logging.getLogger(__name__)


class VoiceTranslationDesktopApp:
    """Desktop application for voice translation"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Voice Translation - Desktop App")
        self.root.geometry("800x600")
        self.root.minsize(600, 400)
        
        # Initialize components
        self.translation_engine: Optional[RealTimeTranslationEngine] = None
        self.audio_processor: Optional[AudioProcessor] = None
        self.is_recording = False
        self.is_offline_mode = True
        
        # Message queue for thread communication
        self.message_queue = queue.Queue()
        
        # Setup UI
        self.setup_ui()
        self.setup_menu()
        
        # Check offline capability
        self.check_offline_capability()
        
        # Start message processing
        self.process_messages()
    
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="Real-Time Voice Translation", 
            font=("Arial", 16, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Language selection frame
        lang_frame = ttk.LabelFrame(main_frame, text="Language Settings", padding="10")
        lang_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        lang_frame.columnconfigure(1, weight=1)
        lang_frame.columnconfigure(3, weight=1)
        
        # Source language
        ttk.Label(lang_frame, text="From:").grid(row=0, column=0, padx=(0, 5))
        self.source_lang_var = tk.StringVar(value="en")
        self.source_lang_combo = ttk.Combobox(
            lang_frame, 
            textvariable=self.source_lang_var,
            values=["English (en)", "Spanish (es)"],
            state="readonly",
            width=15
        )
        self.source_lang_combo.grid(row=0, column=1, padx=(0, 20))
        self.source_lang_combo.set("English (en)")
        
        # Target language
        ttk.Label(lang_frame, text="To:").grid(row=0, column=2, padx=(0, 5))
        self.target_lang_var = tk.StringVar(value="es")
        self.target_lang_combo = ttk.Combobox(
            lang_frame, 
            textvariable=self.target_lang_var,
            values=["Spanish (es)", "English (en)"],
            state="readonly",
            width=15
        )
        self.target_lang_combo.grid(row=0, column=3)
        self.target_lang_combo.set("Spanish (es)")
        
        # Switch languages button
        switch_btn = ttk.Button(
            lang_frame, 
            text="⇄", 
            command=self.switch_languages,
            width=3
        )
        switch_btn.grid(row=0, column=4, padx=(10, 0))
        
        # Translation area
        trans_frame = ttk.LabelFrame(main_frame, text="Translation", padding="10")
        trans_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        trans_frame.columnconfigure(0, weight=1)
        trans_frame.columnconfigure(1, weight=1)
        trans_frame.rowconfigure(1, weight=1)
        
        # Source text
        ttk.Label(trans_frame, text="Source Text:").grid(row=0, column=0, sticky=tk.W)
        self.source_text = tk.Text(trans_frame, height=8, wrap=tk.WORD)
        self.source_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        
        # Target text
        ttk.Label(trans_frame, text="Translated Text:").grid(row=0, column=1, sticky=tk.W)
        self.target_text = tk.Text(trans_frame, height=8, wrap=tk.WORD)
        self.target_text.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        
        # Scrollbars
        source_scrollbar = ttk.Scrollbar(trans_frame, orient=tk.VERTICAL, command=self.source_text.yview)
        source_scrollbar.grid(row=1, column=0, sticky=(tk.E, tk.N, tk.S))
        self.source_text.configure(yscrollcommand=source_scrollbar.set)
        
        target_scrollbar = ttk.Scrollbar(trans_frame, orient=tk.VERTICAL, command=self.target_text.yview)
        target_scrollbar.grid(row=1, column=1, sticky=(tk.E, tk.N, tk.S))
        self.target_text.configure(yscrollcommand=target_scrollbar.set)
        
        # Control buttons frame
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=3, column=0, columnspan=3, pady=(0, 10))
        
        # Voice recording button
        self.record_btn = ttk.Button(
            control_frame,
            text="🎤 Start Recording",
            command=self.toggle_recording,
            style="Accent.TButton"
        )
        self.record_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Translate text button
        translate_btn = ttk.Button(
            control_frame,
            text="Translate Text",
            command=self.translate_text
        )
        translate_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Clear button
        clear_btn = ttk.Button(
            control_frame,
            text="Clear",
            command=self.clear_texts
        )
        clear_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Play audio button
        self.play_btn = ttk.Button(
            control_frame,
            text="🔊 Play Audio",
            command=self.play_translated_audio,
            state=tk.DISABLED
        )
        self.play_btn.pack(side=tk.LEFT)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(
            main_frame, 
            textvariable=self.status_var, 
            relief=tk.SUNKEN, 
            anchor=tk.W
        )
        status_bar.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E))
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            main_frame,
            variable=self.progress_var,
            mode='indeterminate'
        )
        self.progress_bar.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(5, 0))
    
    def setup_menu(self):
        """Setup application menu"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Web Interface", command=self.open_web_interface)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Download Models", command=self.download_models)
        tools_menu.add_command(label="Check Offline Capability", command=self.check_offline_capability)
        tools_menu.add_command(label="Clear Cache", command=self.clear_cache)
        tools_menu.add_separator()
        tools_menu.add_command(label="Translation History", command=self.show_history)
        
        # Settings menu
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        settings_menu.add_checkbutton(label="Offline Mode", variable=tk.BooleanVar(value=True))
        settings_menu.add_command(label="Audio Settings", command=self.audio_settings)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Documentation", command=self.show_documentation)
    
    def check_offline_capability(self):
        """Check offline capability and update status"""
        try:
            capability = offline_manager.check_offline_capability()
            
            if capability["can_work_offline"]:
                self.status_var.set("Offline mode available")
                self.is_offline_mode = True
            else:
                self.status_var.set("Some models not available offline")
                self.is_offline_mode = False
                
                # Show warning
                missing_models = []
                for model_type, models in capability["models"].items():
                    for model_name, available in models.items():
                        if not available:
                            missing_models.append(f"{model_type}/{model_name}")
                
                if missing_models:
                    messagebox.showwarning(
                        "Offline Mode Warning",
                        f"The following models are not available offline:\n" +
                        "\n".join(missing_models) +
                        "\n\nPlease download them using Tools > Download Models"
                    )
            
        except Exception as e:
            logger.error(f"Error checking offline capability: {e}")
            self.status_var.set("Error checking offline capability")
    
    def download_models(self):
        """Download models for offline use"""
        def download_thread():
            try:
                self.status_var.set("Downloading models...")
                self.progress_bar.start()
                
                results = offline_manager.download_all_models()
                
                # Stop progress bar
                self.root.after(0, self.progress_bar.stop)
                
                # Show results
                success_count = sum(1 for success in results.values() if success)
                total_count = len(results)
                
                self.root.after(0, lambda: self.status_var.set(
                    f"Downloaded {success_count}/{total_count} models"
                ))
                
                if success_count == total_count:
                    self.root.after(0, lambda: messagebox.showinfo(
                        "Download Complete",
                        "All models downloaded successfully!"
                    ))
                else:
                    failed_models = [name for name, success in results.items() if not success]
                    self.root.after(0, lambda: messagebox.showwarning(
                        "Download Incomplete",
                        f"Failed to download:\n" + "\n".join(failed_models)
                    ))
                
                # Recheck offline capability
                self.root.after(0, self.check_offline_capability)
                
            except Exception as e:
                logger.error(f"Error downloading models: {e}")
                self.root.after(0, lambda: self.status_var.set("Download failed"))
                self.root.after(0, lambda: messagebox.showerror(
                    "Download Error",
                    f"Failed to download models: {e}"
                ))
                self.root.after(0, self.progress_bar.stop)
        
        # Start download in separate thread
        thread = threading.Thread(target=download_thread, daemon=True)
        thread.start()
    
    def switch_languages(self):
        """Switch source and target languages"""
        source = self.source_lang_var.get()
        target = self.target_lang_var.get()
        
        self.source_lang_var.set(target)
        self.target_lang_var.set(source)
        
        # Update combo boxes
        if "English" in source:
            self.source_lang_combo.set("Spanish (es)")
            self.target_lang_combo.set("English (en)")
        else:
            self.source_lang_combo.set("English (en)")
            self.target_lang_combo.set("Spanish (es)")
    
    def toggle_recording(self):
        """Toggle voice recording"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """Start voice recording"""
        if not self.translation_engine:
            self.initialize_engine()
        
        self.is_recording = True
        self.record_btn.config(text="⏹ Stop Recording")
        self.status_var.set("Recording...")
        
        # Start recording in separate thread
        thread = threading.Thread(target=self.record_audio, daemon=True)
        thread.start()
    
    def stop_recording(self):
        """Stop voice recording"""
        self.is_recording = False
        self.record_btn.config(text="🎤 Start Recording")
        self.status_var.set("Recording stopped")
    
    def record_audio(self):
        """Record audio in separate thread"""
        try:
            # This would integrate with the audio processor
            # For now, just simulate recording
            import time
            time.sleep(2)  # Simulate recording time
            
            # Simulate transcription
            sample_text = "Hello, how are you today?"
            self.root.after(0, lambda: self.source_text.insert(tk.END, sample_text + "\n"))
            
            # Simulate translation
            translated_text = "Hola, ¿cómo estás hoy?"
            self.root.after(0, lambda: self.target_text.insert(tk.END, translated_text + "\n"))
            
        except Exception as e:
            logger.error(f"Error recording audio: {e}")
            self.root.after(0, lambda: self.status_var.set("Recording error"))
    
    def translate_text(self):
        """Translate text from source to target language"""
        source_text = self.source_text.get("1.0", tk.END).strip()
        if not source_text:
            messagebox.showwarning("Warning", "Please enter text to translate")
            return
        
        def translate_thread():
            try:
                self.status_var.set("Translating...")
                self.progress_bar.start()
                
                if not self.translation_engine:
                    self.initialize_engine()
                
                # Get language codes
                source_lang = self.source_lang_var.get().split("(")[1].split(")")[0]
                target_lang = self.target_lang_var.get().split("(")[1].split(")")[0]
                
                # Translate
                result = self.translation_engine.translate_text(
                    source_text, source_lang, target_lang
                )
                
                # Update UI
                self.root.after(0, lambda: self.target_text.delete("1.0", tk.END))
                self.root.after(0, lambda: self.target_text.insert("1.0", result.translated_text))
                self.root.after(0, lambda: self.play_btn.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.status_var.set("Translation complete"))
                self.root.after(0, self.progress_bar.stop)
                
                # Save to history
                translation_data = {
                    "source_text": source_text,
                    "translated_text": result.translated_text,
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "timestamp": str(time.time())
                }
                offline_manager.save_translation_to_history(translation_data)
                
            except Exception as e:
                logger.error(f"Translation error: {e}")
                self.root.after(0, lambda: self.status_var.set("Translation failed"))
                self.root.after(0, lambda: messagebox.showerror("Translation Error", str(e)))
                self.root.after(0, self.progress_bar.stop)
        
        # Start translation in separate thread
        thread = threading.Thread(target=translate_thread, daemon=True)
        thread.start()
    
    def play_translated_audio(self):
        """Play translated audio"""
        try:
            translated_text = self.target_text.get("1.0", tk.END).strip()
            if not translated_text:
                messagebox.showwarning("Warning", "No text to synthesize")
                return
            
            self.status_var.set("Synthesizing audio...")
            
            # This would integrate with TTS
            # For now, just show a message
            messagebox.showinfo("Audio", "Audio synthesis would play here")
            self.status_var.set("Audio played")
            
        except Exception as e:
            logger.error(f"Error playing audio: {e}")
            self.status_var.set("Audio error")
    
    def clear_texts(self):
        """Clear source and target text areas"""
        self.source_text.delete("1.0", tk.END)
        self.target_text.delete("1.0", tk.END)
        self.play_btn.config(state=tk.DISABLED)
        self.status_var.set("Texts cleared")
    
    def initialize_engine(self):
        """Initialize translation engine"""
        try:
            self.translation_engine = RealTimeTranslationEngine(
                device="cpu",  # Use CPU for desktop app
                source_lang="en",
                target_lang="es"
            )
            self.audio_processor = AudioProcessor(
                sample_rate=settings.sample_rate,
                chunk_size=settings.chunk_size
            )
        except Exception as e:
            logger.error(f"Error initializing engine: {e}")
            messagebox.showerror("Initialization Error", f"Failed to initialize: {e}")
    
    def open_web_interface(self):
        """Open web interface in browser"""
        try:
            webbrowser.open("http://localhost:8000")
        except Exception as e:
            logger.error(f"Error opening web interface: {e}")
            messagebox.showerror("Error", "Could not open web interface")
    
    def clear_cache(self):
        """Clear model cache"""
        if messagebox.askyesno("Clear Cache", "Are you sure you want to clear the cache?"):
            offline_manager.clear_cache()
            messagebox.showinfo("Cache Cleared", "Model cache has been cleared")
    
    def show_history(self):
        """Show translation history"""
        history = offline_manager.get_offline_translation_history()
        
        if not history:
            messagebox.showinfo("History", "No translation history found")
            return
        
        # Create history window
        history_window = tk.Toplevel(self.root)
        history_window.title("Translation History")
        history_window.geometry("600x400")
        
        # Create treeview
        tree = ttk.Treeview(history_window, columns=("Source", "Target", "Time"), show="headings")
        tree.heading("Source", text="Source Text")
        tree.heading("Target", text="Translated Text")
        tree.heading("Time", text="Time")
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(history_window, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack widgets
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add history items
        for item in history[-100:]:  # Show last 100 items
            tree.insert("", tk.END, values=(
                item.get("source_text", "")[:50] + "...",
                item.get("translated_text", "")[:50] + "...",
                item.get("timestamp", "")
            ))
    
    def audio_settings(self):
        """Show audio settings dialog"""
        messagebox.showinfo("Audio Settings", "Audio settings dialog would appear here")
    
    def show_about(self):
        """Show about dialog"""
        about_text = f"""
Real-Time Voice Translation Desktop App
Version: {settings.version}

A desktop application for real-time voice translation
between English and Spanish.

Features:
• Offline translation capability
• Voice recording and synthesis
• Translation history
• Web interface integration

Built with Python and Tkinter
        """
        messagebox.showinfo("About", about_text)
    
    def show_documentation(self):
        """Show documentation"""
        try:
            webbrowser.open("https://github.com/your-repo/docs")
        except:
            messagebox.showinfo("Documentation", "Documentation would open in browser")
    
    def process_messages(self):
        """Process messages from other threads"""
        try:
            while True:
                message = self.message_queue.get_nowait()
                # Process message here
                self.message_queue.task_done()
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self.process_messages)


def main():
    """Main entry point for desktop application"""
    root = tk.Tk()
    
    # Set application icon if available
    try:
        root.iconbitmap("app/static/favicon.ico")
    except:
        pass
    
    app = VoiceTranslationDesktopApp(root)
    
    # Start the application
    root.mainloop()


if __name__ == "__main__":
    main() 