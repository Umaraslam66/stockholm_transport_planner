class TransportNode:
    def __init__(self, id, name, type, lat, lon):
        self.id = id
        self.name = name
        self.type = type  # 'bus_stop', 'train_station', 'metro_station'
        self.lat = lat
        self.lon = lon