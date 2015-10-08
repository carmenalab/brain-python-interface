The Database
============

Data generated using this library is either stored in a sqlite3 database using Django's database integration or stored in files whose paths are stored in the database. This makes it easy to run experiments without all the overhead of manually keeping track of which files correspond to which experiment, a procedure which may be prone to human error especially during chaotic experiments. 

Choosing the database type
--------------------------
Django by default supports three different types of databases: sqlite3, mysql and postgreSQL. sqlite3 is the least powerful of these database methodologies, but actually for the purposes here, is the most useful. sqlite3 data is stored in a simple file object, which makes it very portable for multiple users to have their own local copy of the experiment record. The main drawback is of course that only one agent can write to the database or you'll end up with "merge conflicts". But in our case, that's okay, because experiments should only happen in one place. 

Database format
---------------
Django does an excellent job of making database data look like python objects. A database in general may contain many different tables and each table may contain several columns indicating which types of data can be stored in the table. A row in a table is referred to as a *database record*. 

Specifically for this library, the database is the file ``$BMI3D/db/db.sql``. The database contains several tables. All the declared Django models in ``db.tracker.models`` are separate tables in the database. Django converts all the '____Field' attributes of each model into columns of the respective database table. Finally, instances of each type of (e.g., models.TaskEntry instances) correspond to rows in a table (e.g., a row in the TaskEntry table). You interact with these objects in the standard python way, setting values for various ____Field attributes, and Django magically knows how to convert these to and from the database. 
