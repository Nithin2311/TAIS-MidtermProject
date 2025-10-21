# config.py - Configuration settings
class Config:
    """Optimized configuration for midterm submission"""
    
    # Model - Using RoBERTa for better performance
    MODEL_NAME = 'roberta-base'  # 125M parameters
    MAX_LENGTH = 512
    
    # Optimized training parameters
    BATCH_SIZE = 16
    NUM_EPOCHS = 8
    LEARNING_RATE = 2e-5
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    
    # Data splits
    TEST_SIZE = 0.15
    VAL_SIZE = 0.15
    RANDOM_STATE = 42
    
    # Paths
    DATA_PATH = 'data/raw/Resume.csv'
    MODEL_SAVE_PATH = 'models/resume_classifier_midterm'
    
    @staticmethod
    def display_config():
        """Display configuration for presentation"""
        print("="*70)
        print("📋 PROJECT CONFIGURATION")
        print("="*70)
        print(f"Model: {Config.MODEL_NAME}")
        print(f"Max Length: {Config.MAX_LENGTH} tokens")
        print(f"Batch Size: {Config.BATCH_SIZE}")
        print(f"Epochs: {Config.NUM_EPOCHS}")
        print(f"Learning Rate: {Config.LEARNING_RATE}")
        print(f"Train/Val/Test Split: 70%/15%/15%")
        print("="*70 + "\n")
