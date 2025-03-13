class Indicator:
	def __init__(self, trend, description):
		self.trend = trend
		self.description = description
		self.dates = []

	def add_date(self, d):
		self.dates.append(d)