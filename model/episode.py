
class Episode(object):

    def __init__(self, csv_line):
        self.id                     = csv_line[0]
        self.title                  = csv_line[1]
        self.original_air_date      = csv_line[2]
        self.production_code        = csv_line[3]
        self.season                 = csv_line[4]
        self.number_in_season       = csv_line[5]
        self.number_in_series       = csv_line[6]
        self.us_viewers_in_millions = csv_line[7]
        self.views                  = csv_line[8]
        self.imdb_rating            = csv_line[9]
        self.imdb_votes             = csv_line[10]
        self.image_url              = csv_line[11]
        self.video_url              = csv_line[12]