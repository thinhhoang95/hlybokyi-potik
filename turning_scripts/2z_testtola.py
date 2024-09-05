from get_tola import tola_slope_detector 
import numpy as np

def test_detect_takeoff():
    # Case 1: Takeoff
    h = np.array([500, 700, 1200, 1300, 1300, 1200, 1100, 1200, 1300, 1400, 1500, 2000, 2200, 2300, 2500, 2700])
    t = np.arange(len(h))
    print('Case 1: ', tola_slope_detector(t, h))

    # Case 2: Yet takeoff, yet landing
    h = np.array([500, 700, 800])
    t = np.arange(len(h))
    print('Case 2: ', tola_slope_detector(t, h))

    # Case 3: Took off, yet landing
    h = np.array([6000, 4000, 3000, 3000, 3000, 3000, 3000])
    t = np.arange(len(h))
    print('Case 3: ', tola_slope_detector(t, h))

    # Case 4: Took off, landing
    h = np.array([5000, 4000, 3000, 2000, 1500, 1300, 1200, 1100, 1000, 900, 800, 700, 600, 500, 400, 300, 200, 100, 0])
    t = np.arange(len(h))
    print('Case 4: ', tola_slope_detector(t, h), ' len = ', len(h)) # something's wrong here

    # Case 5: Take off and landing
    h = np.array([500, 700, 1200, 1300, 1300, 1500, 2000, 2200, 2100, 2000, 1900, 1800, 1700, 1600, 1500, 1400, 1300, 1200, 1100, 1000, 900, 800, 700, 600, 500, 400, 300, 200, 100, 0])
    t = np.arange(len(h))
    print('Case 5: ', tola_slope_detector(t, h), ' len = ', len(h))

    # Case 6: cruise
    h = np.array([3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000])
    t = np.arange(len(h))
    print('Case 6: ', tola_slope_detector(t, h), ' len = ', len(h))
    

if __name__ == "__main__":
    test_detect_takeoff()