QUnit.test("Features manipulation", function(assert) {
  var feats = new Features();
  // clear to start
  feats.clear();
  assert.equal($.isEmptyObject(feats.get_checked_features()), true);

  // check 1 feature
  feats.select_features(["feature1"])
  sel_feats = feats.get_checked_features();
  assert.equal(Object.keys(sel_feats).length, 1);
  assert.equal(sel_feats["feature1"], true);

  // test callback on change
  var feats_callback_var = 0;
  feats.bind_change_callback(function() {feats_callback_var += 1;});
  $("#feat_feature1").trigger("click");
  $("#feat_feature1").trigger("click");
  assert.equal(feats_callback_var, 2);

  // if you disable the features, the callback should not trigger
  feats.disable_entry();
  $("#feat_feature1").trigger("click");
  assert.equal(feats_callback_var, 2);  

  feats.unbind_change_callback();
  $("#feat_feature1").trigger("click");
  assert.equal(feats_callback_var, 2); // no change because disabled and unbound  

  // clear again
  feats.clear();
  feats.enable_entry();
  $("#feat_feature1").trigger("click");
  assert.equal(feats_callback_var, 2);  // no increment because callback should be unbound

  sel_feats = feats.get_checked_features();
  assert.equal(Object.keys(sel_feats).length, 1);
  assert.equal(sel_feats["feature1"], true);
});