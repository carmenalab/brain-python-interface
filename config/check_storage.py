import os
def main():
	b = os.path.isdir('/storage/')
	return b

if __name__=='__main__':
	b = main()
	print(b)
