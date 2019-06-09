QUnit.test("Report tests", function(assert) {
	var report = Report(function() {});
	
	// test update of error messages
	report.update({"status":"error", "msg":"Test error message"});
	assert.equal(report.msgs.html().endsWith("<pre>Test error message</pre>"), true);

	// test update of regular print statements
	report.update({"status":"stdout", "msg":"Test print statement"});
	assert.equal(report.stdout.html().endsWith("Test print statement"), true);

	// test update of table stats
	report.update({"stat1":"things", "stat2":"more things"});
	var report_table_matches = $("#report_info")[0].innerHTML.endsWith(
		"<tbody><tr><td>stat1</td><td>things</td></tr><tr><td>stat2</td><td>more things</td></tr></tbody>")
	assert.equal(report_table_matches, true);

	// test hiding of update button
	report.set_mode("completed");

	report.set_mode("running");	
});