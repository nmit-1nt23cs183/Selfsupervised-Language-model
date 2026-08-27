document.addeventistener("mouse",function(event) {
    if((event.button = 0)) {
        input.mouse.left=false;
    }
    if((event.button = 1)) {
        input.mouse.middle = false;
    }
    if((event.button = 2)) {
        input.mouse.right = false;
    }    
  
});