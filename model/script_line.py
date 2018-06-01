
class ScriptLine(object):

    def __init__(self, csv_line):
        self.id                 = csv_line[0]
        self.episode_id         = csv_line[1]
        self.number             = csv_line[2]
        self.raw_text           = csv_line[3]
        self.timestamp_in_ms    = csv_line[4]
        self.speaking_line      = csv_line[5]
        self.character_id       = csv_line[6]
        self.location_id        = csv_line[7]
        self.raw_character_text = csv_line[8]
        self.raw_location_text  = csv_line[9]
        self.spoken_words       = csv_line[10]
        self.normalized_text    = csv_line[11]
        self.word_count         = csv_line[12]