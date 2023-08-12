function performSearch() {
    const videoName = $("#videoSelector").val();
    const searchTerm = $("#searchInput").val();
    
    const button = $("button");
    button.prop("disabled", true);
    button.text("Loading...");

    const url = `https://inference-server-mlsd-video-search.darkube.app/${videoName}?search_entry=${searchTerm}`;
    
    $.getJSON(url, function(data) {
        $("#results").empty();

        data.forEach(item => {
            let image = `<img src="data:image/png;base64,${item.image_base64}" alt="Scene Image"/>`;
            let time = secondsToHMS(item.second);  // Convert seconds to the format hours:minutes:seconds
            let movieInfo = `<h3>${item.video_name}</h3><span>${time}</span>`;
            let resultItem = `<div class="result-item">${image}${movieInfo}</div>`;
            
            $("#results").append(resultItem);
        });

        button.text("Search");
        button.prop("disabled", false);
    })
    .fail(function() {
        alert('Error occurred during the search.');
        button.text("Search");
        button.prop("disabled", false);
    });
}

function secondsToHMS(seconds) {
    const hours = Math.floor(seconds / 3600);
    seconds %= 3600;
    const minutes = Math.floor(seconds / 60);
    seconds %= 60;
    
    return `${hours}h ${minutes}m ${seconds}s`;
}
