Starting tasks through the web interface
========================================
[This document still in progress]

Architecture
------------
To allow various interface and peripheral devices to operate asynchronously, a single task started through the web interface is actually comprised of many processes. For example, below might be an example of a task + features. Processes always span in the tree structure shown on the left and tend to communicate with each other in the directions shown (though all communication links are bidirectional). 

	..	image:: processes.png

In this codebase, asynchronous computation tends to happen in separate processes. In the existing functionality, there is not much need for processes to share the same memory space. The main reason for keeping things asynchronous is that the main task event loop must execute repeatedly in ~15 ms, which may be too restrictive for some tasks to fully finish execution. This means threads and processes tend to be functionally equivalent for the existing functionality. 

	..	image:: webserver_setup.png

The main mechanisms for inter-process communication (IPC) are pipes (common alternatives include shared memory, sockets, files, etc.). In normal operation, the main "parent" process is the webserver (we will treat it as one process for this discussion). Spawning another process in which to run the task allows the webserver to continue responding to browser events while the task runs independently (e.g., so you can look at old tasks and figure out what to do next). 

Sometimes the side tasks requested through the web browser are totally independent and don't require any communication, but you still want to be able to do them without freezing the browser. An example might include training a BMI decoder, which might take a few minutes but does not require user/peripheral device input. In that case, spawning the very specific "task" process is overkill since no IPC is needed. In these cases, the celery package provides functionality for simply wrapping a function and having it run asynchronously. An example of this can be seen in the file db/trainbmi.py, in which the functions cache_plx and make_bmi have been marked as celery tasks through the use of the @tasks() decorators. This adds a method '.si' to the function (yes, it is strange for a function to have methods) which are used by the function cache_and_train to run the tasks asynchronously. 

