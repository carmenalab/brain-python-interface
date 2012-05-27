function Report() {
    this.obj = $(document.createElement("div"));
    $("#report").append(this.obj);
}
Report.prototype.update = function(info) {
    this.obj.html("");
    if (info.state && info.state == "error") {
        this.obj.html("<h1>Error</h1><pre>"+info.msg+"</pre>");
    }
}
Report.prototype.destroy = function () {
    this.obj.remove();
}