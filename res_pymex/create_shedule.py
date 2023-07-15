"""Create Schedule for CMG simulator."""
from datetime import datetime
from dateutil.relativedelta import relativedelta

def generate_schedule(start:int, stop:int, file_name:str):
    """Generate a *DATE for each month of the schedule.inc"""
    data_inicio = datetime.strptime(f'{start} 01 01', '%Y %m %d')
    data_fim = datetime.strptime(f'{stop+1} 01 01', '%Y %m %d')
    data_atual = data_inicio
    with open(file_name, 'w', encoding='UTF-8') as file:
        while data_fim >= data_atual:
            data_str = data_atual.strftime("%Y %m %d")
            file.write(f'*DATE {data_str}\n')
            if data_atual.month == 12:
                file.write('\n')
            data_atual += relativedelta(months=1)


if __name__ == '__main__':
    generate_schedule(2016, 2036, file_name='SCHEDULE.inc')
