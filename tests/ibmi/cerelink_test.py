if __name__ == "__main__":
    import csv
    import time
    import argparse
    parser = argparse.ArgumentParser(description="Collects spike data for a set amount of time")
    parser.add_argument("output", help="Output csv file")
    args = parser.parse_args()

    with open(args.output, "w") as f:
        csvfile = csv.DictWriter(f, SpikeEventData._fields)
        csvfile.writeheader()

        channels = [5, 6, 7, 8]

        conn = Connection()
        conn.connect()
        conn.select_channels(channels)
        conn.start_data()

        gen = conn.get_event_data()

        got_first = False

        start = time.time()
        while (time.time()-start) < 3:
            spike_event_data = gen.next()
            if not got_first and spike_event_data is not None:
                print spike_event_data
                got_first = True

            if spike_event_data is not None:
                csvfile.writerow(dict(spike_event_data._asdict()))

        conn.stop_data()
        conn.disconnect()