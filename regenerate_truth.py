from config import Config
print("Initializing Config...")
c = Config()
print("Config initialized. Checking for truth files...")
import os
if os.path.exists(c.data_path + '/dev/ref/truth-' + c.DATASET_ROOT + '.txt'):
    print("✅ Dev truth file generated.")
else:
    print("❌ Dev truth file NOT generated.")
