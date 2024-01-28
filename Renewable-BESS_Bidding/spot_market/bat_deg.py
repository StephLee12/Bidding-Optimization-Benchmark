import numpy as np
import rainflow

"""
Based on Bolun's work: Modeling of Lithium-Ion Battery Degradation for Cell Life Assessment
"""


class BatteryDegradation:
    def __init__(self,update_interval) -> None:
        # coeff settings
        self.ktime = 4.14e-10  # per second
        self.ksoc = 1.04
        self.soc_ref = 0.50
        self.ktemp = 6.93e-2
        self.temp_ref = 293  # Kelvin
        self.kDelta1 = 1.40e5
        self.kDelta2 = -5.01e-1
        self.kDelta3 = -1.23e5

        # SoC profile length 
        self.update_interval = update_interval

    # get remaining capacity 
    def cal_rem_cap(self,soc_lst,temp_lst,timeslot,cur_battery_life):
        self.count_battery_cycle(soc_lst)
        # calendar_deg = self.cal_calendar_deg(np.mean(soc_lst),np.mean(temp_lst),timeslot)
        calendar_deg = 0
        cycle_deg = self.cal_cycle_deg(temp_lst,timeslot)

        update_battery_life = 1 - (1-cur_battery_life) * np.exp(-(calendar_deg+cycle_deg*0.1))

        return update_battery_life

    # rainflow 
    def count_battery_cycle(self,soc_lst):
        rng_lst, mean_lst, count_lst, idx_start_lst, idx_end_lst = [], [], [], [], []
        for rng, mean, count, idx_start, idx_end in rainflow.extract_cycles(soc_lst):
            rng_lst.append(rng)
            mean_lst.append(mean)
            count_lst.append(count)
            idx_start_lst.append(idx_start)
            idx_end_lst.append(idx_end)
        
        self.rng_lst = rng_lst
        self.mean_lst = mean_lst
        self.count_lst = count_lst
        self.idx_start_lst = idx_start_lst
        self.idx_end_lst = idx_end_lst

    # calendar degradation 
    def cal_calendar_deg(self,avg_soc,avg_temp,timeslot):
        temp_stress = self.temp_stress(avg_temp)
        time_stress = self.time_stress(timeslot)
        soc_stress = self.soc_stress(avg_soc)

        return temp_stress * time_stress * soc_stress

    # cycle degradation
    def cal_cycle_deg(self,temp_lst,timeslot):
        timeslot = timeslot - self.update_interval
        cycle_deg = 0
        for rng,avg_soc,count,idx_start,idx_end in zip(self.rng_lst,self.mean_lst,self.count_lst,self.idx_start_lst,self.idx_end_lst):
            # DoD stress 
            dod = rng * 2
            dod_stress = self.dod_stress(dod)
            # time stress 
            # time = timeslot + idx_end
            time_stress = self.time_stress(idx_end-idx_start)
            # soc stress 
            soc_stress = self.soc_stress(avg_soc)
            # temp stress 
            avg_temp= np.mean(temp_lst[idx_start:idx_end+1])
            temp_stress = self.temp_stress(avg_temp)

            cycle_deg += count * (max(0,dod_stress) + time_stress) * soc_stress * temp_stress
        
        return cycle_deg

    # temperature stress 
    def temp_stress(self,temp):
        return np.exp(self.ktemp*(temp-self.temp_ref)*(self.temp_ref/temp))

    # soc stress
    def soc_stress(self,soc):
        return np.exp(self.ksoc*(soc-self.soc_ref))

    # time stress 
    def time_stress(self,timeslot):
        '''
        the unit of timeslot is 5 min
        '''
        timeslot = timeslot * 60 * 5 
        return self.ktime * timeslot

    def dod_stress(self,dod):
        return 1/(self.kDelta1*np.power(dod,self.kDelta2)+self.kDelta3) if dod != 0 else 0