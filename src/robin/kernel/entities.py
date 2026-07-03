"""Entities for the kernel module."""

import datetime
import numpy as np
import pandas as pd
import random
import os

from robin.demand.entities import Demand, Passenger
from robin.kernel.constants import OUTPUT_PATH
from robin.kernel.exceptions import TicketNotBoughtException
from robin.supply.constants import MAX_DEPTH
from robin.supply.entities import Service, Supply

from loguru import logger
from pathlib import Path
from typing import Dict, List, Union


class Kernel:
    """
    The kernel class integrates the supply and demand modules.

    Attributes:
        supply (Supply): Supply object.
        demand (Demand): Demand object.
        passengers (List[Passenger]): List of passengers.
        passengers_purchase_date (Dict[datetime.date, List[Passenger]]): Dictionary with passengers grouped by purchase date.
        simulation_days (List[datetime.date]): List of simulation days.
        simulation_day_idx (int): Index of the current simulation day.
    """
    
    def __init__(self, path_config_supply: str, path_config_demand: str, seed: Union[int, None] = None) -> None:
        """
        Initialize a kernel object.

        Args:
            path_config_supply (str): Path to the supply configuration file.
            path_config_demand (str): Path to the demand configuration file.
        """
        if seed is not None:
            self.set_seed(seed)
        self.supply = Supply.from_yaml(path_config_supply)
        self.demand = Demand.from_yaml(path_config_demand)
        self.passengers = self.demand.generate_passengers()
        self.passengers_purchase_date = self._group_passengers_by_purchase_date(self.passengers)
        self.simulation_days = self._get_simulation_days()
        self._simulation_day_idx = 0

    def _get_simulation_days(self) -> List[datetime.date]:
        """
        Get the simulation days (purchase dates of the passengers).
        
        Returns:
            List[datetime.date]: List of simulation days.
        """
        simulation_days = list(self.passengers_purchase_date.keys())
        simulation_days.sort()
        return simulation_days

    def _group_passengers_by_purchase_date(self, passengers: List[Passenger]) -> Dict[datetime.date, List[Passenger]]:
        """
        Group passengers by purchase day.
        
        Args:
            passengers (List[Passenger]): List of passengers.
        
        Returns:
            List[List[Passenger]]: List of passengers grouped by purchase day.
        """
        passengers_purchase_date = {}
        for passenger in passengers:
            purchase_date = passenger.purchase_date
            if purchase_date in passengers_purchase_date:
                # Append the passenger to the list
                passengers_purchase_date[purchase_date].append(passenger)
            else:
                # Create a new list with the passenger
                passengers_purchase_date[purchase_date] = [passenger]
        return passengers_purchase_date

    def _simulate(
        self,
        passengers: List[Passenger],
        output_path: Union[Path, None] = None,
        max_depth: int = MAX_DEPTH,
        departure_time_hard_restriction: bool = True
     ) -> None:
        """
        Private method to simulate the demand-supply interaction.
        
        Args:
            passengers (List[Passenger]): List of passengers.
            output_path (Path, optional): Path to the output csv file. Defaults to None.
            max_depth (int, optional): Maximum depth for the journey search algorithm. Defaults to MAX_DEPTH.
            departure_time_hard_restriction (bool, optional): If True, the passenger will not
                be assigned to a service with a departure time that is not valid. Defaults to True.
        """
        for passenger in passengers:
            # Filter services by passenger's origin-destination and date
            journeys = self.supply.filter_journeys(
                origin=passenger.market.departure_station,
                destination=passenger.market.arrival_station,
                date=passenger.arrival_day.date,
                max_depth=max_depth
            )

            # Calculate utility for each journey
            journey_arg_max = {
                'journey': None,
                'seats': None,
                'utility': 0,
                'ticket_price': 0
            }

            for journey in journeys:
                utility, seats = passenger.get_utility(
                    journey=journey,
                    departure_time_hard_restriction=departure_time_hard_restriction
                )

                # Update journey with max utility available
                if utility > journey_arg_max['utility']:
                    journey_arg_max['journey'] = journey
                    journey_arg_max['seats'] = seats['seats']
                    journey_arg_max['utility'] = utility
                    journey_arg_max['ticket_price'] = sum(seats['price'])
            
            # Buy ticket if utility is positive
            if journey_arg_max['utility'] > 0:
                for i, service in enumerate(journey_arg_max['journey'].services):
                    origin, destination = journey_arg_max['journey'].markets[service]
                    ticket_bought = service.buy_ticket(
                        origin=origin,
                        destination=destination,
                        seat=journey_arg_max['seats'][i],
                        purchase_date=passenger.purchase_date
                    )
                    if not ticket_bought:
                        raise TicketNotBoughtException(service)
                passenger.journey = journey_arg_max['journey']
                passenger.seats = journey_arg_max['seats']
                passenger.ticket_price = journey_arg_max['ticket_price']
                passenger.utility = journey_arg_max['utility']

        # Save passengers data to csv file
        if output_path is not None:
            self._to_csv(passengers, output_path)

    def _to_csv(self, passengers: List[Passenger], output_path: str = OUTPUT_PATH) -> None:
        """
        Save passengers data to CSV file.

        Args:
            passengers (List[Passenger]): List of passengers.
            output_path (str, optional): Path to the output CSV file. Defaults to 'output.csv'.
        """
        column_names = [
            'id', 'user_pattern', 'departure_station', 'arrival_station',
            'arrival_day', 'arrival_time', 'purchase_date', 'service', 'service_departure_time',
            'service_arrival_time', 'seat', 'price', 'utility'
        ]
        data = []
        for passenger in passengers:
            details = {
                'services': [service.id for service in passenger.journey.services] if passenger.journey else None,
                'departure_time': passenger.journey.departure_time if passenger.journey else None,
                'arrival_time': passenger.journey.arrival_time if passenger.journey else None,
                'seats': [seat.name for seat in passenger.seats] if passenger.journey else None,
                'utility': passenger.utility if passenger.journey else None
            }
            data.append([
                passenger.id,
                passenger.user_pattern,
                passenger.market.departure_station,
                passenger.market.arrival_station,
                passenger.arrival_day,
                passenger.arrival_time,
                passenger.purchase_date,
                details['services'],
                details['departure_time'],
                details['arrival_time'],
                details['seats'],
                passenger.ticket_price,
                details['utility']
            ])
        df = pd.DataFrame(data=data, columns=column_names)
        df.to_csv(output_path, index=False)

    @property
    def simulation_day(self) -> datetime.date:
        """
        Get the current simulation day.

        Returns:
            datetime.date: Current simulation day.
        """
        return self.simulation_days[self._simulation_day_idx]

    @property
    def is_simulation_finished(self) -> bool:
        """
        Check if the simulation is finished.

        Returns:
            bool: True if the simulation is finished, False otherwise.
        """
        return self._simulation_day_idx >= len(self.simulation_days)

    def filter_supply_by_tsp(self, tsp_id: str) -> Supply:
        """
        Filters the supply object by TSP and creates a new supply object.

        Args:
            tsp (TSP): TSP object.

        Returns:
            Supply: Supply object filtered by TSP.
        """
        services = self.supply.filter_services_by_tsp(tsp_id=tsp_id)
        supply = Supply(services=services)
        return supply

    def simulate(
        self,
        output_path: Union[Path, None] = None,
        max_depth: int = MAX_DEPTH,
        departure_time_hard_restriction: bool = True
    ) -> List[Service]:
        """
        Simulate the demand-supply interaction.

        The passengers will maximize the utility for each service and seat, according to
        its origin-destination and date, buying a ticket only if the utility is positive.

        Args:
            output_path (str, optional): Path to the output CSV file. Defaults to None.
            max_depth (int, optional): Maximum depth for the journey search algorithm. Defaults to MAX_DEPTH.
            departure_time_hard_restriction (bool, optional): If True, the passenger will not
                be assigned to a service with a departure time that is not valid. Defaults to True.

        Returns:
            List[Service]: List of services with updated tickets.
        """
        self._simulate(
            passengers=self.passengers,
            output_path=output_path,
            max_depth=max_depth,
            departure_time_hard_restriction=departure_time_hard_restriction
        )
        self._simulation_day_idx = len(self.simulation_days)
        return self.supply.services
  
    def simulate_a_day(
        self,
        output_path: Union[Path, None] = None,
        max_depth: int = MAX_DEPTH,
        departure_time_hard_restriction: bool = True
    ) -> List[Service]:
        """
        Simulate the demand-supply interaction for a day.
        
        The difference with the simulate method is that this method will only simulate the first available purchase day
        of the passengers. This method is useful to simulate the demand-supply interaction for a day, and then
        modify the supply object and simulate again the next day, for example, for a RL environment.

        Args:
            output_path (Path, optional): Path to the output csv file. Defaults to None.
            max_depth (int, optional): Maximum depth for the journey search algorithm. Defaults to MAX_DEPTH.
            departure_time_hard_restriction (bool, optional): If True, the passenger will not
                be assigned to a service with a departure time that is not valid. Defaults to True.

        Returns:
            List[Service]: List of services with updated tickets.
        """
        # Check if all days have been simulated
        if self.is_simulation_finished:
            logger.warning('All days have been simulated, simulation will not continue.')
            return self.supply.services
        # Simulate demand-supply interaction for the next available purchase day
        self._simulate(
            passengers=self.passengers_purchase_date[self.simulation_day],
            output_path=output_path,
            max_depth=max_depth,
            departure_time_hard_restriction=departure_time_hard_restriction
        )
        # Update simulation day index
        self._simulation_day_idx += 1
        return self.supply.services

    def set_seed(self, seed: int) -> None:
        """
        Set seed for the random number generator.

        Args:
            seed (int): Seed for the random number generator.
        """
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
