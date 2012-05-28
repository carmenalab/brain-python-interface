function Report(active, callback) {
    this.callback = callback;
    this.obj = document.createElement("div");
    this.info = document.createElement("table");
    this.msgs = document.createElement("div");
    this.info.className = "options";
    this.msgs.className = "rightside";
    this.obj.appendChild(this.info);
    this.obj.appendChild(this.msgs);
    $("#report").append(this.obj);

    this.boxes = {"state":"Current State", "trials":"Trial #", "length":"Time", "reward_len":"Reward Time"};
    for (var i in this.boxes) {
        var row = document.createElement("tr");
        var label = document.createElement("td");
        label.innerHTML = this.boxes[i];
        row.appendChild(label);
        var data = document.createElement("td");
        row.appendChild(data);
        this.info.appendChild(row);
        this.boxes[i] = data;
    }
    
    if (active) {
        this.ws = new WebSocket("ws://"+hostname.split(":")[0]+":8001/connect");
        this.ws.onmessage = function(evt) {
            this.update(JSON.parse(evt.data));
        }.bind(this);
    }
}
Report.prototype.update = function(info) {
    if (typeof(this.callback) == "function" && info)
        this.callback(info);
    if (info.status && info.status == "error") {
        this.msgs.innerHTML = "<pre>"+info.msg+"</pre>";
    } else {
        for (var i in this.boxes) {
            if (info[i])
                this.boxes[i].innerHTML = info[i];
        }
    }
}
Report.prototype.destroy = function () {
    this.obj.parentNode.removeChild(this.obj);
    if (this.ws)
        this.ws.close();
}