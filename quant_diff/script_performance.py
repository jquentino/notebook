import numpy as np
import matplotlib.pyplot as mplt
import pyaccel as pa
import pymodels as pm
import time
def comp_perf(acc, bun, nturns=100, nsamples=300):
    times = np.zeros(nsamples)
    for i in range(nsamples):
        print(f'{i:03d}/{nsamples:03d}', end='\r')
        t0 = time.process_time()
        part_out, *_ = pa.tracking.ring_pass(
            acc, bun, nr_turns=nturns, turn_by_turn=True, parallel=False)
        times[i] = time.process_time() - t0
    print('', end='\r')
    return times.mean()*1e3, times.std()*1e3
def compute_performance(npart=1, nturns=100, nsamples=300):
    acc = pm.si.create_accelerator()
    acc.cavity_on=True
    acc.vchamber_on = True
    co = pa.tracking.find_orbit6(acc)
    bun = np.zeros([6, npart])
    bun += co
    bun[0] += 1e-3
    bun[2] += 1e-6
    tmpl = '{0:30s} {1:.2f}+-{2:.2f}ms'
    # Without rad effects
    pa.utils.set_random_seed(1029847)
    acc.radiation_on = 'off'
    print(tmpl.format(
        'Without Radiation',
        *comp_perf(acc, bun, nturns, nsamples)))
    # Only radiation damping
    pa.utils.set_random_seed(1029847)
    acc.radiation_on = 'damping'
    print(tmpl.format(
        'Damping',
        *comp_perf(acc, bun, nturns, nsamples)))
    # Damping and quantum diffusion (uniform distribution)
    pa.utils.set_distribution('uniform')
    pa.utils.set_random_seed(1029847)
    acc.radiation_on='full'
    print(tmpl.format(
        'Damping + diffusion (uniform)',
        *comp_perf(acc, bun, nturns, nsamples)))
    # Damping and quantum diffusion (normal distribution)
    pa.utils.set_distribution('normal')
    pa.utils.set_random_seed(1029847)
    acc.radiation_on='full'
    print(tmpl.format(
        'Damping + diffusion (normal)',
        *comp_perf(acc, bun, nturns, nsamples)))
def save_data(filename, nturns=500, radiation='damping'):
    acc = pm.si.create_accelerator()
    acc.cavity_on=True
    acc.vchamber_on = True
    co = pa.tracking.find_orbit6(acc)
    bun = np.zeros([6, 1])
    bun += co
    bun[0] += 1e-3
    bun[2] += 1e-6
    # Damping and quantum diffusion (normal distribution)
    pa.utils.set_distribution('normal')
    pa.utils.set_random_seed(1029847)
    acc.radiation_on = radiation
    part_out, *_ = pa.tracking.ring_pass(
        acc, bun, nr_turns=nturns, turn_by_turn=True, parallel=False)
    np.save(filename, part_out)
compute_performance(npart=1, nturns=500, nsamples=10)
# save_data('original_full', nturns=500, radiation='full')
# save_data('modified_full', nturns=500, radiation='full')
