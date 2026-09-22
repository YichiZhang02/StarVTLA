import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from scipy.spatial.transform import Rotation

from vtla.datasets.tcp_contract import validate_tcp_contract
from vtla.engine.utils.ee_transforms import encode_relative_tcp


def dataset(root, robot_type, names, states, actions):
    data = root / 'data/chunk-000'
    episodes = root / 'meta/episodes/chunk-000'
    data.mkdir(parents=True)
    episodes.mkdir(parents=True)
    features = {key: {'dtype': 'float32', 'shape': [len(names)], 'names': names}
                for key in ['observation.state', 'action']}
    # Legacy derived fields must be removed from all metadata and parquet schemas.
    features['action_absolute_quat'] = {'dtype': 'float32', 'shape': [8], 'names': ['q'] * 8}
    info = {'robot_type': robot_type, 'features': features, 'fps': 30}
    (root / 'meta/info.json').write_text(json.dumps(info))
    (root / 'meta/stats.json').write_text(json.dumps({'action_relative_quat': {'mean': [0]}}))
    n = len(states)
    table = pa.table({'episode_index': [0] * n, 'frame_index': list(range(n)),
                      'observation.state': states.tolist(), 'action': actions.tolist(),
                      'action_absolute_quat': [[0.] * 8] * n})
    pq.write_table(table, data / 'file-000.parquet')
    pq.write_table(pa.table({'episode_index': [0], 'stats/action_absolute_quat/mean': [[0.] * 8]}),
                   episodes / 'file-000.parquet')


def verify(root):
    info = json.loads((root / 'meta/info.json').read_text())
    validate_tcp_contract(info['tcp_contract'], offsets=[1, 2], robot_type=info['robot_type'])
    stats = json.loads((root / 'meta/stats.json').read_text())
    table = pq.read_table(root / 'data/chunk-000/file-000.parquet')
    assert not any('quat' in key for key in info['features'])
    assert not any('quat' in key for key in stats)
    assert not any('quat' in key for key in table.column_names)
    S = torch.tensor(table['observation.state_absolute_ee'].to_pylist())
    A = torch.tensor(table['action_absolute_ee'].to_pylist())
    expected = torch.cat([encode_relative_tcp(S[:-k], A[k:], S.shape[-1] // 10) for k in (1, 2)])
    np.testing.assert_allclose(stats['action_relative_ee']['mean'], expected.mean(0).numpy(), atol=1e-6)
    ep = pq.read_table(root / 'meta/episodes/chunk-000/file-000.parquet')
    np.testing.assert_allclose(ep['stats/action_relative_ee/mean'][0].as_py(), stats['action_relative_ee']['mean'], atol=1e-6)
    assert not any('quat' in key for key in ep.column_names)
    return S, A


def test_joint_converter_and_online_fk_match(tmp_path, monkeypatch):
    from tools import convert_joints_to_eepose as converter
    from vtla.engine.utils.ee_kinematics import joint_indices, to_absolute_ee, flange_to_tcp
    from deployment.robots import RobotConfig
    class Algo:
        def rm_algo_forward_kinematics(self, joints, flag=0):
            q = Rotation.from_euler('z', joints[0], degrees=True).as_quat()
            return [0.2, -0.1, 0.3, q[3], *q[:3]]
    names = [f'left_main_joint{i}' for i in range(1, 8)] + ['left_main_gripper']
    S = np.zeros((4, 8)); S[:, 0] = [0.1, 0.2, 0.3, 0.4]; S[:, 7] = 0.5
    A = S.copy(); A[:, 0] += 0.1
    dataset(tmp_path, 'rm_isf_umi_left', names, S, A)
    monkeypatch.setattr(converter, 'make_realman_algo', lambda force_type: Algo())
    monkeypatch.setattr('sys.argv', ['convert', '--root', str(tmp_path), '--horizon', '2', '--action-gap', '1'])
    converter.main()
    actual, _ = verify(tmp_path)
    calibration = {'left': RobotConfig.get_flange_tcp_calibration('rm_isf_umi_left', 'left')}
    expected = np.stack([to_absolute_ee(Algo(), row, joint_indices(names), flange_tcp_calibration=calibration) for row in S])
    np.testing.assert_allclose(actual.numpy(), expected, atol=1e-7)
    rotation = Rotation.from_euler('z', S[0, 0]).as_matrix()
    tcp_pos, _ = flange_to_tcp(np.array([0.2, -0.1, 0.3]), rotation, *calibration['left'])
    np.testing.assert_allclose(actual[0, :3], tcp_pos, atol=1e-7)


def test_umi_converter_retains_tcp_and_migration_preserves_source(tmp_path, monkeypatch):
    from tools import migrate_tcp_dataset
    names = [f'{side}_{key}' for side in ('left', 'right') for key in ('x','y','z','qx','qy','qz','qw','gripper')]
    S = np.tile(np.array([0.2, 0.1, 0.3, 0, 0, 0, 1, 0.5] * 2), (4, 1))
    A = S.copy(); A[:, 0] += 0.01
    src, dst = tmp_path / 'source', tmp_path / 'converted'
    dataset(src, 'umi', names, S, A)
    source_data = src / 'data/chunk-000/file-000.parquet'
    table = pq.read_table(source_data)
    calibrated = np.tile([0., 0., 0., 1., 0., 0., 0., 1., 0., .31] * 2, (4, 1))
    for feature in ('observation.state_absolute_ee', 'action_absolute_ee'):
        table = table.append_column(feature, pa.array(calibrated.tolist()))
    pq.write_table(table, source_data)
    before = (src / 'meta/info.json').read_bytes()
    argv = ['migrate', '--src', str(src), '--dst', str(dst), '--horizon', '2', '--action-gap', '1']
    monkeypatch.setattr('sys.argv', argv + ['--dry-run'])
    migrate_tcp_dataset.main()
    assert not dst.exists()
    monkeypatch.setattr('sys.argv', argv)
    migrate_tcp_dataset.main()
    actual, _ = verify(dst)
    np.testing.assert_allclose(actual[0, :3], S[0, 8:11], atol=1e-7)
    np.testing.assert_allclose(actual[0, 10:13], S[0, :3], atol=1e-7)
    np.testing.assert_allclose(actual[:, [9, 19]], .31, atol=1e-7)
    assert (src / 'meta/info.json').read_bytes() == before
    monkeypatch.setattr('sys.argv', argv)
    with pytest.raises(SystemExit):
        migrate_tcp_dataset.main()


def test_signed_stats_offsets_and_gripper_rebuild(tmp_path, monkeypatch):
    from tools import convert_umi_to_eepose, normalize_episode_grippers
    from tools.rebuild_relative_ee_stats import rebuild
    from vtla.datasets.tcp_stats import collect_relative_stats
    names = [f'{side}_{key}' for side in ('left', 'right') for key in ('x','y','z','qx','qy','qz','qw','gripper')]
    S = np.tile(np.array([0.2, 0.1, 0.3, 0, 0, 0, 1, 0.5] * 2), (4, 1))
    S[:, 7] = [.1, .3, .5, .7]; S[:, 15] = [.2, .4, .6, .8]
    A = S.copy(); A[:, 0] += .01
    dataset(tmp_path, 'umi', names, S, A)
    monkeypatch.setattr('sys.argv', ['convert', '--root', str(tmp_path), '--horizon', '2', '--action-gap', '1'])
    convert_umi_to_eepose.main()
    rebuild(tmp_path, 3, -1)
    info = json.loads((tmp_path / 'meta/info.json').read_text())
    validate_tcp_contract(info['tcp_contract'], offsets=[-1, 0, 1])
    paths = list((tmp_path / 'data').rglob('*.parquet'))
    stats, _ = collect_relative_stats(paths, [-1, 0, 1])
    assert stats['count'] == [10]
    monkeypatch.setattr('sys.argv', ['normalize', '--root', str(tmp_path), '--min', '0.5', '--max', '1'])
    normalize_episode_grippers.main()
    stats, eps = collect_relative_stats(paths, [-1, 0, 1])
    saved = json.loads((tmp_path / 'meta/stats.json').read_text())
    np.testing.assert_allclose(saved['action_relative_ee']['mean'], stats['mean'])
    ep = pq.read_table(tmp_path / 'meta/episodes/chunk-000/file-000.parquet')
    np.testing.assert_allclose(ep['stats/action_relative_ee/mean'][0].as_py(), eps[0]['mean'])
    assert stats['min'][9] >= .5
    assert stats['max'][19] <= 1


def test_metadata_creation_persists_contracts_and_positional_root(tmp_path):
    from vtla.datasets.dataset_metadata import LeRobotDatasetMetadata
    from vtla.datasets.tcp_contract import build_tcp_contract
    from vtla.datasets.visual_preprocess import make_visual_preprocess
    root = tmp_path / 'metadata'
    contract = build_tcp_contract('umi', 2, 1)
    visual = make_visual_preprocess(size=224, wrist_undistort=True, tactile_encoding=None)
    meta = LeRobotDatasetMetadata.create('test', 30, {}, 'umi', root, False,
                                        tcp_contract=contract, visual_preprocess=visual)
    stored = json.loads((root / 'meta/info.json').read_text())
    assert stored['tcp_contract'] == meta.tcp_contract == contract
    assert stored['visual_preprocess'] == meta.visual_preprocess == visual
