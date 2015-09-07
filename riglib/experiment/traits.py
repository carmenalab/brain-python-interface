try:
    import traits.api as traits
except ImportError:
    import enthought.traits.api as traits

Float = traits.Float
Int = traits.Int
Tuple = traits.Tuple
Array = traits.Array
String = traits.String
Bool = traits.Bool
Instance = traits.Instance

class Enum(traits.Enum):
	pass

class DataFile(traits.Instance):
	def __init__(self, *args, **kwargs):
		if 'bmi3d_query_kwargs' in kwargs:
			self.bmi3d_query_kwargs = kwargs['bmi3d_query_kwargs']

		super(DataFile, self).__init__(*args, **kwargs)
