from datetime import datetime  
from datetime import timedelta



def get_delta_minute(num_min):
    return timedelta(seconds=60*num_min)

def add_delta_to_time(tm, dt):
    return tm + dt

def get_list_date_increment(start_date, end_date, increment_min):
    current_date = start_date
    date_list = []
    dt = get_delta_minute(increment_min)
    while current_date <= end_date:
        date_list.append(current_date)
        current_date = add_delta_to_time(current_date, dt)
    return date_list

def convert_datetime_list_to_string(dt_list):
    return [datetime.strftime(i, '%Y-%m-%d %H:%M') for i in dt_list]

if __name__ == "__main__":
    start_date = datetime(2020, 5, 17, 0, 0)
    end_date = datetime(2020, 5, 20, 0, 0)
    
    dt_list = get_list_date_increment(start_date, end_date, 20)
    l = convert_datetime_list_to_string(dt_list)

    print(l)






