import time


def get_timeslot(period, slot,time):
    """
    Passing
        slot = 0 will return current time slot;
        slot = -1 will return the last time slot;
        slot = -2 will return the second last time slot;
        slot = 1 will return the next time slot;
        slot = 2 will return the time slot next to the next time slot;
        etc
    """
    # now = int(time.time())
    now = int(time)
    period = period.lower()
    if period == "day":
        num_of_seconds = 24 * 60 * 60
    elif period == "week":
        num_of_seconds = 24 * 60 * 60 * 7
    elif period == "month":
        num_of_seconds = 24 * 60 * 60 * 30
    else:
        raise ValueError("Invalid period")

    return (now - now % num_of_seconds + slot * num_of_seconds)

# 1627197501     到   1627283902
print(get_timeslot("week", 0,1627197501))
# 1626912000
print(get_timeslot("week", 0,1627283902))
# 86,401    24 * 60 * 60=