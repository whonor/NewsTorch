import os
import sys
import subprocess
import logging
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import SelectField, SubmitField, IntegerField
from wtforms.validators import DataRequired, NumberRange
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'

# Define supported models and datasets
SUPPORTED_MODELS = [
    'NRMS', 'NAML', 'LSTUR', 'NPA', 'DKN', 'FIM', 'TANR', 
    'CENNEWSREC', 'MINS', 'IPNR', 'CNE-SUE', 'LKPNR'
]

SUPPORTED_DATASETS = [
    'MIND-small', 'MIND-large', 'ebnerd_demo', 'ebnerd_small'
]

DATASET_DOWNLOAD_SCRIPTS = {
    'MIND-small': 'dataset_download_prepare/download_mind.py',
    'MIND-large': 'dataset_download_prepare/download_mind.py',
    'ebnerd_demo': 'dataset_download_prepare/download_ebnerd.py',
    'ebnerd_small': 'dataset_download_prepare/download_ebnerd.py'
}

DATASET_PREPARE_SCRIPTS = {
    'MIND-small': 'dataset_download_prepare/MIND_dataset_prepare.py',
    'MIND-large': 'dataset_download_prepare/MIND_dataset_prepare.py',
    'ebnerd_demo': 'dataset_download_prepare/EBNeRD_dataset_prepare.py',
    'ebnerd_small': 'dataset_download_prepare/EBNeRD_dataset_prepare.py'
}

class ModelSelectionForm(FlaskForm):
    model = SelectField('Select Model:', choices=[(model, model) for model in SUPPORTED_MODELS], validators=[DataRequired()])
    dataset = SelectField('Select Dataset:', choices=[(dataset, dataset) for dataset in SUPPORTED_DATASETS], validators=[DataRequired()])
    batch_size = IntegerField('Batch Size:', default=64, validators=[DataRequired(), NumberRange(min=1)])
    epoch = IntegerField('Epoch:', default=10, validators=[DataRequired(), NumberRange(min=1)])
    dataset_root = SelectField('Dataset Root (DATASET_ROOT):', choices=[
        ('MIND-small', 'MIND-small'),
        ('MIND-large', 'MIND-large'),
        ('ebnerd_demo', 'ebnerd_demo'),
        ('ebnerd_small', 'ebnerd_small')
    ], default='MIND-small')
    dataset_name = SelectField('Dataset Name (dataset_name):', choices=[
        ('MIND', 'MIND'),
        ('ebnerd', 'ebnerd')
    ], default='MIND')
    dataset_size = SelectField('Dataset Size (dataset_size):', choices=[
        ('small', 'small'),
        ('large', 'large'),
        ('demo', 'demo'),
        ('submission', 'submission')
    ], default='small')
    submit = SubmitField('Start Training')

def validate_model_dataset(model):
    """Validate if the model is supported"""
    # Check if model is supported
    if model not in SUPPORTED_MODELS:
        return False, f"Unsupported model: {model}"
    
    return True, ""

@app.route('/', methods=['GET', 'POST'])
def index():
    form = ModelSelectionForm()
    if form.validate_on_submit():
        model = form.model.data
        batch_size = form.batch_size.data
        epoch = form.epoch.data
        dataset_root = form.dataset_root.data
        dataset_name = form.dataset_name.data
        dataset_size = form.dataset_size.data
        # Redirect to training page, passing model and dataset parameters
        return redirect(url_for('train', model=model, batch_size=batch_size, epoch=epoch, 
                                dataset_root=dataset_root, dataset_name=dataset_name, dataset_size=dataset_size))
    return render_template('index.html', form=form)

@app.route('/train')
def train():
    try:
        model = request.args.get('model')
        batch_size = request.args.get('batch_size', 64, type=int)
        epoch = request.args.get('epoch', 10, type=int)
        dataset_root = request.args.get('dataset_root', 'MIND-small')
        dataset_name = request.args.get('dataset_name', 'MIND')
        dataset_size = request.args.get('dataset_size', 'small')
        
        # Validate model
        is_valid, error_message = validate_model_dataset(model)
        if not is_valid:
            return jsonify({
                'status': 'error',
                'message': error_message
            }), 400
        
        # Check if dataset files exist
        data_name = dataset_root
        dataset_name_check = dataset_name
        train_root = f"./{data_name}/train"
        dev_root = f"./{data_name}/dev"
        test_root = f"./{data_name}/test"
        
        required_files = [
            f"{train_root}/news.tsv",
            f"{train_root}/behaviors.tsv",
            f"{dev_root}/news.tsv",
            f"{dev_root}/behaviors.tsv",
            f"{test_root}/news.tsv",
            f"{test_root}/behaviors.tsv"
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            return jsonify({
                'status': 'error',
                'message': f"Missing required dataset files: {', '.join(missing_files)}. Please download and process the dataset first."
            }), 400
        
        # Build training command
        cmd = [
            sys.executable, 'general_runner.py',
            '--model', model,
            '--batch_size', str(batch_size),
            '--epoch', str(epoch),
            '--mode', 'train',
            '--DATASET_ROOT', dataset_root,
            '--dataset_name', dataset_name,
            '--dataset_size', dataset_size
        ]
        
        # Set environment variables
        env = os.environ.copy()
        
        # Run training command in background
        subprocess.Popen(cmd, env=env)
        
        return jsonify({
            'status': 'started',
            'message': f'Training task started: Model {model}, Batch Size {batch_size}, Epoch {epoch}, DATASET_ROOT {dataset_root}, dataset_name {dataset_name}, dataset_size {dataset_size}',
            'model': model,
            'batch_size': batch_size,
            'epoch': epoch,
            'dataset_root': dataset_root,
            'dataset_name': dataset_name,
            'dataset_size': dataset_size
        })
    except Exception as e:
        logger.error(f"Training task failed to start: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Training task failed to start: {str(e)}'
        }), 500

@app.route('/dev')
def dev():
    try:
        model = request.args.get('model')
        dev_model_path = request.args.get('dev_model_path')
        dataset_root = request.args.get('dataset_root', 'MIND-small')
        dataset_name = request.args.get('dataset_name', 'MIND')
        dataset_size = request.args.get('dataset_size', 'small')
        
        # Validate model
        is_valid, error_message = validate_model_dataset(model)
        if not is_valid:
            return jsonify({
                'status': 'error',
                'message': error_message
            }), 400
        
        # Check if model file exists
        if not os.path.exists(dev_model_path):
            return jsonify({
                'status': 'error',
                'message': f"Model file does not exist: {dev_model_path}"
            }), 400
        
        # Check if dataset files exist
        data_name = dataset_root
        dataset_name_check = dataset_name
        train_root = f"./{data_name}/train"
        dev_root = f"./{data_name}/dev"
        test_root = f"./{data_name}/test"
        
        required_files = [
            f"{train_root}/news.tsv",
            f"{train_root}/behaviors.tsv",
            f"{dev_root}/news.tsv",
            f"{dev_root}/behaviors.tsv",
            f"{test_root}/news.tsv",
            f"{test_root}/behaviors.tsv"
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            return jsonify({
                'status': 'error',
                'message': f"Missing required dataset files: {', '.join(missing_files)}. Please download and process the dataset first."
            }), 400
        
        # Build validation command
        cmd = [
            sys.executable, 'general_runner.py',
            '--model', model,
            '--mode', 'dev',
            '--dev_model_path', dev_model_path,
            '--DATASET_ROOT', dataset_root,
            '--dataset_name', dataset_name,
            '--dataset_size', dataset_size
        ]
        
        # Set environment variables
        env = os.environ.copy()
        
        # Run validation command in background
        subprocess.Popen(cmd, env=env)
        
        return jsonify({
            'status': 'started',
            'message': f'Validation task started: Model {model}, DATASET_ROOT {dataset_root}, dataset_name {dataset_name}, dataset_size {dataset_size}',
            'model': model,
            'dataset_root': dataset_root,
            'dataset_name': dataset_name,
            'dataset_size': dataset_size
        })
    except Exception as e:
        logger.error(f"Validation task failed to start: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Validation task failed to start: {str(e)}'
        }), 500

@app.route('/test')
def test():
    try:
        model = request.args.get('model')
        test_model_path = request.args.get('test_model_path')
        dataset_root = request.args.get('dataset_root', 'MIND-small')
        dataset_name = request.args.get('dataset_name', 'MIND')
        dataset_size = request.args.get('dataset_size', 'small')
        
        # Validate model
        is_valid, error_message = validate_model_dataset(model)
        if not is_valid:
            return jsonify({
                'status': 'error',
                'message': error_message
            }), 400
        
        # Check if model file exists
        if not os.path.exists(test_model_path):
            return jsonify({
                'status': 'error',
                'message': f"Model file does not exist: {test_model_path}"
            }), 400
        
        # Check if dataset files exist
        data_name = dataset_root
        dataset_name_check = dataset_name
        train_root = f"./{data_name}/train"
        dev_root = f"./{data_name}/dev"
        test_root = f"./{data_name}/test"
        
        required_files = [
            f"{train_root}/news.tsv",
            f"{train_root}/behaviors.tsv",
            f"{dev_root}/news.tsv",
            f"{dev_root}/behaviors.tsv",
            f"{test_root}/news.tsv",
            f"{test_root}/behaviors.tsv"
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            return jsonify({
                'status': 'error',
                'message': f"Missing required dataset files: {', '.join(missing_files)}. Please download and process the dataset first."
            }), 400
        
        # Build test command
        cmd = [
            sys.executable, 'general_runner.py',
            '--model', model,
            '--mode', 'test',
            '--test_model_path', test_model_path,
            '--DATASET_ROOT', dataset_root,
            '--dataset_name', dataset_name,
            '--dataset_size', dataset_size
        ]
        
        # Set environment variables
        env = os.environ.copy()
        
        # Run test command in background
        subprocess.Popen(cmd, env=env)
        
        return jsonify({
            'status': 'started',
            'message': f'Test task started: Model {model}, DATASET_ROOT {dataset_root}, dataset_name {dataset_name}, dataset_size {dataset_size}',
            'model': model,
            'dataset_root': dataset_root,
            'dataset_name': dataset_name,
            'dataset_size': dataset_size
        })
    except Exception as e:
        logger.error(f"Test task failed to start: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Test task failed to start: {str(e)}'
        }), 500

@app.route('/download_dataset')
def download_dataset():
    try:
        dataset = request.args.get('dataset')
        dataset_root = request.args.get('dataset_root', 'MIND-small')
        dataset_name = request.args.get('dataset_name', 'MIND')
        dataset_size = request.args.get('dataset_size', 'small')
        
        # Check if dataset is supported
        if dataset not in DATASET_DOWNLOAD_SCRIPTS:
            return jsonify({
                'status': 'error',
                'message': f"Unsupported dataset: {dataset}"
            }), 400
        
        # Get download script path
        script_path = DATASET_DOWNLOAD_SCRIPTS[dataset]
        
        # Check if script exists
        if not os.path.exists(script_path):
            return jsonify({
                'status': 'error',
                'message': f"Download script does not exist: {script_path}"
            }), 400
        
        # Build download command
        cmd = [sys.executable, script_path]
        
        # Set environment variables
        env = os.environ.copy()
        
        # Run download command in background
        subprocess.Popen(cmd, env=env)
        
        return jsonify({
            'status': 'started',
            'message': f'Dataset {dataset} download task started. Please check console output for detailed progress. DATASET_ROOT: {dataset_root}, dataset_name: {dataset_name}, dataset_size: {dataset_size}'
        })
    except Exception as e:
        logger.error(f"Dataset download task failed to start: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Dataset download task failed to start: {str(e)}'
        }), 500

@app.route('/prepare_dataset')
def prepare_dataset():
    try:
        dataset = request.args.get('dataset')
        dataset_root = request.args.get('dataset_root', 'MIND-small')
        dataset_name = request.args.get('dataset_name', 'MIND')
        dataset_size = request.args.get('dataset_size', 'small')
        
        # Check if dataset is supported
        if dataset not in DATASET_PREPARE_SCRIPTS:
            return jsonify({
                'status': 'error',
                'message': f"Unsupported dataset: {dataset}"
            }), 400
        
        # Get processing script path
        script_path = DATASET_PREPARE_SCRIPTS[dataset]
        
        # Check if script exists
        if not os.path.exists(script_path):
            return jsonify({
                'status': 'error',
                'message': f"Processing script does not exist: {script_path}"
            }), 400
        
        # Build processing command
        cmd = [sys.executable, script_path]
        
        # Set environment variables
        env = os.environ.copy()
        
        # Run processing command in background
        subprocess.Popen(cmd, env=env)
        
        return jsonify({
            'status': 'started',
            'message': f'Dataset {dataset} processing task started. Please check console output for detailed progress. DATASET_ROOT: {dataset_root}, dataset_name: {dataset_name}, dataset_size: {dataset_size}'
        })
    except Exception as e:
        logger.error(f"Dataset processing task failed to start: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Dataset processing task failed to start: {str(e)}'
        }), 500

if __name__ == '__main__':
    # Ensure templates directory exists
    if not os.path.exists('templates'):
        os.makedirs('templates')
    app.run(debug=True)

