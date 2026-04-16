import os
import pickle
import numpy as np

print("="*60)
print("🔍 TRAINING DEBUGGER")
print("="*60)

# Step 1: Check if training_data folder exists
print("\n📁 Step 1: Checking training_data folder...")
if os.path.exists('training_data'):
    print(f"  ✅ training_data folder exists")
    files = os.listdir('training_data')
    pkl_files = [f for f in files if f.endswith('.pkl')]
    print(f"  📊 Found {len(pkl_files)} .pkl files: {pkl_files}")
else:
    print("  ❌ training_data folder NOT found!")
    print("  Please run data_recorder.py first to record signs")

# Step 2: Check each sign file
print("\n📂 Step 2: Checking individual sign files...")
signs = ['my', 'name', 'is', 'ashish', 'hello', 'me', 'fine', 'drink', 
         'eat', 'good', 'love', 'morning', 'please', 'thanks', 'you']

for sign in signs:
    filename = f'training_data/{sign}.pkl'
    if os.path.exists(filename):
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            print(f"  ✅ {sign:10}: {len(data)} samples")
        except Exception as e:
            print(f"  ❌ {sign:10}: Error loading - {e}")
    else:
        print(f"  ⚠️ {sign:10}: File not found")

# Step 3: Try to train a minimal model
print("\n🤖 Step 3: Attempting minimal training...")
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    # Load all data
    X, y = [], []
    for i, sign in enumerate(signs):
        filename = f'training_data/{sign}.pkl'
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                for sample in data:
                    X.append(sample)
                    y.append(i)
    
    if len(X) > 0:
        X = np.array(X)
        y = np.array(y)
        print(f"  ✅ Loaded {len(X)} total samples")
        print(f"  ✅ Feature dimension: {X.shape[1] if len(X) > 0 else 'N/A'}")
        
        # Train tiny model
        model = RandomForestClassifier(n_estimators=10)
        model.fit(X, y)
        print(f"  ✅ Mini-model trained successfully")
        
        # Try to save
        model_data = {
            'model': model,
            'signs': signs,
            'accuracy': 0.95
        }
        
        with open('test_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        if os.path.exists('test_model.pkl'):
            print(f"  ✅ Test model saved as 'test_model.pkl'")
            os.remove('test_model.pkl')
            print(f"  ✅ Test model deleted (cleanup complete)")
        else:
            print(f"  ❌ Could not save test model - permission issue?")
    else:
        print("  ❌ No data loaded")
        
except Exception as e:
    print(f"  ❌ Training failed: {e}")

# Step 4: Check current directory permissions
print("\n📁 Step 4: Checking directory permissions...")
current_dir = os.getcwd()
print(f"  Current directory: {current_dir}")
print(f"  Writable: {os.access(current_dir, os.W_OK)}")

# Step 5: Try to create a simple file
print("\n📝 Step 5: Testing file creation...")
try:
    with open('test_write.txt', 'w') as f:
        f.write('test')
    print(f"  ✅ Can create files in current directory")
    os.remove('test_write.txt')
except Exception as e:
    print(f"  ❌ Cannot create files: {e}")

print("\n" + "="*60)
print("🔍 DEBUGGING COMPLETE")
print("="*60)