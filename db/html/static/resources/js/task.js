function Task(id) {
    this.task = id;
    if (typeof(id) == "number")
        this.task = $.get("ajax/exp_info/"+id);

    //Wipe out parameters and sequence
    
}