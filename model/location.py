
class Location(object):

    def __init__(self, csv_line):
        self.id                 = csv_line[0]
        self.name               = csv_line[1]
        self.normalized_name    = csv_line[2]
