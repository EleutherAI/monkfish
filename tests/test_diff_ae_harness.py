import pytest
import jax
import jax.numpy as jnp
import optax
from unittest.mock import MagicMock, patch, call
import monkfish.lvd.diffusion_ae as dae

@pytest.fixture
def mock_cfg():
    return {
        "gcp": {"gcp_credentials_path": "path/to/credentials", "gcp_bucket_name": "test-bucket"},
        "diffusion_auto_encoder": {
            "data_loader": {"fs_type": "local", "data_root_directory": "/tmp/data"},
            "checkpoints": {"fs_type": "local", "ckpt_root_directory": "/tmp/checkpoints"},
            "dist_manager": {"mesh_shape": (1, 1)},
            "model": {
                "encoder": {"k": 16, "n_layers": 4},
                "decoder": {"k": 16, "n_layers": 4}
            },
            "train": {"lr": 0.001, "ckpt_freq": 1000, "total_steps": 10000, "log_freq": 100}
        },
        "seed": 42
    }

@pytest.fixture
def mock_args():
    args = MagicMock()
    args.operation = "train_dae"
    return args

@pytest.fixture
def harness(mock_cfg, mock_args):
    with patch('catfish.lvd.diffusion_ae.sdl'), patch('catfish.lvd.diffusion_ae.du'), patch('catfish.lvd.diffusion_ae.daed'):
        return dae.DiffAEHarness(mock_args, mock_cfg)

def test_parse_args(harness):
    with patch.object(harness, 'parse_args') as mock_parse:
        harness.parse_args()
        mock_parse.assert_called_once()

@pytest.mark.parametrize("fs_type, expected_class", [
    ("local", "os_filesystem"),
    ("gcp", "gcp_filesystem"),
])
def test_init_fs(harness, fs_type, expected_class):
    harness.cfg["diffusion_auto_encoder"]["checkpoints"]["fs_type"] = fs_type
    with patch(f'your_module.sdl.{expected_class}') as mock_fs:
        harness.init_fs()
        mock_fs.assert_called_once()

@pytest.mark.parametrize("operation", ["train_dae", "autoencode"])
def test_init_data_loader(harness, operation):
    harness.args.operation = operation
    with patch('catfish.lvd.diffusion_ae.sdl.ShardedDataDownloader') as mock_downloader:
        harness.init_data_loader()
        mock_downloader.assert_called_once()

def test_init_dist_manager(harness):
    with patch('catfish.lvd.diffusion_ae.du.DistManager') as mock_dist_manager:
        harness.init_dist_manager()
        mock_dist_manager.assert_called_once()

def test_make_model_prng_key_splitting(harness):
    initial_key = jax.random.PRNGKey(0)
    harness.state['prng_key'] = initial_key
    
    with patch('catfish.lvd.diffusion_ae.daed.Encoder'), patch('catfish.lvd.diffusion_ae.daed.Decoder'):
        harness.make_model()
    
    assert not jnp.array_equal(harness.state['prng_key'], initial_key)

def test_make_optimizer_initialization(harness):
    mock_model = MagicMock()
    harness.state['model'] = mock_model
    
    with patch('optax.adam') as mock_adam:
        harness.make_optimizer()
    
    mock_adam.assert_called_once()
    assert 'opt_state' in harness.state

@patch('catfish.lvd.diffusion_ae.sdl.os_filesystem')
def test_save_checkpoint_creates_directories(mock_os_fs, harness):
    harness.save_checkpoint(1000)
    mock_os_fs.return_value.makedirs.assert_has_calls([
        call('/ckpt_1000', recreate=True),
        call('/ckpt_1000/model/encoder', recreate=True),
        call('/ckpt_1000/model/decoder', recreate=True),
        call('/ckpt_1000/opt_state', recreate=True)
    ], any_order=True)

@patch('catfish.lvd.diffusion_ae.sdl.os_filesystem')
def test_load_checkpoint_nonexistent(mock_os_fs, harness):
    mock_os_fs.return_value.listdir.return_value = []
    
    with pytest.raises(ValueError, match="No checkpoint found to load."):
        harness.load_checkpoint()

def test_most_recent_ckpt_empty(harness):
    with patch.object(harness, '_list_checkpoints', return_value=[]):
        assert harness.most_recent_ckpt() == 0

@pytest.mark.parametrize("checkpoints,expected", [
    (['ckpt_1000', 'ckpt_2000'], 2000),
    (['ckpt_500', 'ckpt_1500', 'ckpt_1000'], 1500),
    (['ckpt_100'], 100),
])
def test_most_recent_ckpt_various(harness, checkpoints, expected):
    with patch.object(harness, '_list_checkpoints', return_value=checkpoints):
        assert harness.most_recent_ckpt() == expected

def test_new_ckpt_path_increment(harness):
    harness.cfg['diffusion_auto_encoder']['train']['ckpt_freq'] = 500
    with patch.object(harness, 'most_recent_ckpt', return_value=1000):
        assert harness.new_ckpt_path(1500) == '/tmp/checkpoints/ckpt_1500'

def test_new_ckpt_path_mismatch(harness):
    harness.cfg['diffusion_auto_encoder']['train']['ckpt_freq'] = 500
    with patch.object(harness, 'most_recent_ckpt', return_value=1000):
        with pytest.raises(AssertionError):
            harness.new_ckpt_path(2000)  # Should be 1500

def test_latest_ckpt_path_empty(harness):
    with patch.object(harness, '_list_checkpoints', return_value=[]):
        assert harness.latest_ckpt_path() is None

@patch('catfish.lvd.diffusion_ae.sdl.ShardedDataDownloader')
@patch('catfish.lvd.diffusion_ae.dc')
def test_train_interruption(mock_dc, mock_downloader, harness):
    mock_downloader.return_value.step.side_effect = KeyboardInterrupt()
    
    harness.train()
    
    mock_downloader.return_value.stop.assert_called_once()

@patch('catfish.lvd.diffusion_ae.sdl.ShardedDataDownloader')
@patch('catfish.lvd.diffusion_ae.dc')
def test_train_logging(mock_dc, mock_downloader, harness, capsys):
    mock_downloader.return_value.step.return_value = jnp.zeros((32, 32, 3))
    mock_dc.update_state_dict.return_value = (0.5, {})
    harness.cfg['diffusion_auto_encoder']['train']['log_freq'] = 5
    harness.cfg['diffusion_auto_encoder']['train']['total_steps'] = 10
    
    harness.train()
    
    captured = capsys.readouterr()
    assert "Step 5, Average Loss:" in captured.out
    assert "Step 10, Average Loss:" in captured.out

@patch('catfish.lvd.diffusion_ae.sdl.ShardedDataDownloader')
@patch('catfish.lvd.diffusion_ae.dc')
def test_train_checkpointing(mock_dc, mock_downloader, harness):
    mock_downloader.return_value.step.return_value = jnp.zeros((32, 32, 3))
    mock_dc.update_state_dict.return_value = (0.5, {})
    harness.cfg['diffusion_auto_encoder']['train']['ckpt_freq'] = 5
    harness.cfg['diffusion_auto_encoder']['train']['total_steps'] = 10
    
    with patch.object(harness, 'save_checkpoint') as mock_save:
        harness.train()
    
    mock_save.assert_has_calls([call(5), call(10)])

def test_autoencode_no_checkpoints(harness):
    with patch.object(harness, 'latest_ckpt_path', return_value=None):
        with pytest.raises(ValueError, match="No checkpoint found to load."):
            harness.autoencode()

def test_init(harness):
    assert harness.cfg is not None
    assert harness.state is not None
    assert harness.optimizer is not None
    assert harness.ckpt_fs is not None

def test_make_model(harness):
    assert 'model' in harness.state
    assert 'encoder' in harness.state['model']._fields
    assert 'decoder' in harness.state['model']._fields

def test_make_optimizer(harness):
    assert isinstance(harness.optimizer, optax.GradientTransformation)
    assert 'opt_state' in harness.state

@patch('catfish.lvd.diffusion_ae.sdl.os_filesystem')
def test_list_checkpoints(mock_os_fs, harness):
    mock_os_fs.return_value.listdir.return_value = ['ckpt_1000', 'ckpt_2000', 'other_dir']
    mock_os_fs.return_value.isdir.side_effect = lambda x: x.startswith('ckpt_')
    
    checkpoints = harness._list_checkpoints()
    assert checkpoints == ['ckpt_1000', 'ckpt_2000']

@patch('catfish.lvd.diffusion_ae.sdl.os_filesystem')
def test_save_checkpoint(mock_os_fs, harness):
    harness.state['prng_key'] = jax.random.PRNGKey(0)
    harness.save_checkpoint(1000)
    
    mock_os_fs.return_value.makedirs.assert_called()
    # Add more assertions to check if all components are saved

@patch('catfish.lvd.diffusion_ae.sdl.os_filesystem')
def test_load_checkpoint(mock_os_fs, harness):
    mock_os_fs.return_value.listdir.return_value = ['ckpt_1000']
    mock_os_fs.return_value.isdir.return_value = True
    
    with patch.object(harness.state['model'].encoder, 'load'), \
         patch.object(harness.state['model'].decoder, 'load'), \
         patch.object(harness.dist_manager, 'load_array'):
        harness.load_checkpoint()
    
    # Add assertions to check if all components are loaded correctly

def test_most_recent_ckpt(harness):
    with patch.object(harness, '_list_checkpoints', return_value=['ckpt_1000', 'ckpt_2000']):
        assert harness.most_recent_ckpt() == 2000

def test_new_ckpt_path(harness):
    harness.cfg['diffusion_auto_encoder']['train']['ckpt_freq'] = 1000
    with patch.object(harness, 'most_recent_ckpt', return_value=1000):
        assert harness.new_ckpt_path(2000) == '/tmp/checkpoints/ckpt_2000'

def test_latest_ckpt_path(harness):
    with patch.object(harness, '_list_checkpoints', return_value=['ckpt_1000', 'ckpt_2000']):
        assert harness.latest_ckpt_path() == '/ckpt_2000'

@patch('catfish.lvd.diffusion_ae.sdl.ShardedDataDownloader')
@patch('catfish.lvd.diffusion_ae.dc')
def test_train(mock_dc, mock_downloader, harness):
    mock_downloader.return_value.step.return_value = jnp.zeros((32, 32, 3))
    mock_dc.update_state_dict.return_value = (0.5, {})
    
    harness.train()
    
    assert mock_downloader.return_value.start.called
    assert mock_downloader.return_value.step.called
    assert mock_dc.update_state_dict.called
    assert mock_downloader.return_value.ack.called
    assert mock_downloader.return_value.stop.called

def test_autoencode(harness):
    with patch.object(harness, 'latest_ckpt_path', return_value='/ckpt_2000'), \
         patch.object(harness, 'load_checkpoint') as mock_load:
        harness.autoencode()
        mock_load.assert_called_with('/ckpt_2000')