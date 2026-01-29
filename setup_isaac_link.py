import os

# Isaac Sim paths extracted from sys.path
isaac_paths = [
    '/home/lfw/isaacsim/kit/python/lib/python3.11/site-packages',
    '/home/lfw/isaacsim/python_packages',
    '/home/lfw/isaacsim/exts/isaacsim.simulation_app',
    '/home/lfw/isaacsim/extsDeprecated/omni.isaac.kit',
    '/home/lfw/isaacsim/kit/kernel/py',
    '/home/lfw/isaacsim/kit/plugins/bindings-python',
    '/home/lfw/isaacsim/exts/isaacsim.robot_motion.lula/pip_prebundle',
    '/home/lfw/isaacsim/exts/isaacsim.asset.exporter.urdf/pip_prebundle',
    '/home/lfw/isaacsim/extscache/omni.kit.pip_archive-0.0.0+69cbf6ad.lx64.cp311/pip_prebundle',
    '/home/lfw/isaacsim/exts/omni.isaac.core_archive/pip_prebundle',
    '/home/lfw/isaacsim/exts/omni.isaac.ml_archive/pip_prebundle',
    '/home/lfw/isaacsim/exts/omni.pip.compute/pip_prebundle',
    '/home/lfw/isaacsim/exts/omni.pip.cloud/pip_prebundle'
]

venv_site_packages = os.path.abspath(".venv/lib/python3.11/site-packages")
pth_file = os.path.join(venv_site_packages, 'isaac_sim.pth')

if not os.path.exists(venv_site_packages):
    print(f"Error: {venv_site_packages} does not exist.")
    exit(1)

with open(pth_file, 'w') as f:
    for p in isaac_paths:
        f.write(p + '\n')

print(f"Created {pth_file}")
