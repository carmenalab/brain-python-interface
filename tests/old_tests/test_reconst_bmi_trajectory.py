import unittest
import dbfunctions as dbfn
from tasks.point_mass_cursor import CursorPlantWithMass

class TestDecoderTraining(unittest.TestCase):
    # point mass block:
    te = dbfn.TaskEntry(3138)
    hdf = te.hdf
    cutoff = 1000;
    cursor_pos = hdf.root.task[:cutoff]['cursor_pos']
    cursor_vel = hdf.root.task[:cutoff]['cursor_vel']

    spike_counts = hdf.root.task[:cutoff]['spike_counts']
    spike_counts = np.array(spike_counts, dtype=np.float64)
    
    internal_state = hdf.root.task[:cutoff]['internal_decoder_state']
    update_bmi_ix = np.nonzero(np.diff(np.squeeze(internal_state[:, 9, 0])))[0]+1

    plant = CursorPlantWithMass(endpt_bounds=(-14, 14, 0., 0., -14, 14))

    def run_decoder(dec, spike_counts, update_bmi_ix):
        T = spike_counts.shape[0]
        decoded_state = []
        spike_accum = np.zeros_like(spike_counts[0,:])
        dec_last = np.zeros_like(dec.predict(spike_counts[0,:]))

        for t in range(T):
            spike_accum = spike_accum+spike_counts[t,:]
            if t in update_bmi_ix:
                dec_new = dec.predict(spike_accum)
                decoded_state.append(dec_new)
                dec_last = dec_new
                spike_accum = np.zeros_like(spike_counts[0,:])
            else:
                decoded_state.append(dec_last)
        return np.array(np.vstack(decoded_state))
    
    dec = te.decoder;
    dec_state_mm = run_decoder(dec, spike_counts, update_bmi_ix)

    def move_plant(dec_state_mn, cursor_pos, cursor_vel, plant, dt = 1/60.):
        p0 = cursor_pos[0,:]
        v0 = cursor_vel[0,:]
        
        pos_arr = []
        vel_arr = []

        for i in range(1, dec_state_mn.shape[0]):
            force = dec_state_mn[i,[9, 10, 11]]
            vel = v0 + dt*force
            pos = p0 + dt*vel + 0.5*dt**2*force
            
            pos, vel = plant._bound(pos,vel)

            p0 = pos.copy()
            v0 = vel.copy()

            pos_arr.append(pos)
            vel_arr.append(vel)
        return np.array(pos_arr), np.array(vel_arr)

    pos, vel = move_plant(dec_state_mm, cursor_pos, cursor_vel)


    diff_mm = cursor - np.float32(dec_state_mm[:,0:3])
    self.assertEqual(np.max(np.abs(diff_mm)), 0)
    
    dec = dbfunctions.get_decoder(te)
    dec_cm = train.rescale_KFDecoder_units(dec)
    dec_state_cm = run_decoder(dec_cm, spike_counts)
    diff_cm = cursor - np.float32(dec_state_cm[:,0:3])
    #print np.max(np.abs(diff_cm))
    self.assertEqual(np.max(np.abs(diff_cm)), 0)
