namespace ImageClassificationPractice
{
    public static class Helpers
    {
        public static IEnumerable<ImageData> LoadImagesFromDirectory(string folder, bool useFolderNameAsLabel)
        {
            string[] files = Directory.GetFiles(folder, "*", searchOption: SearchOption.AllDirectories);
            foreach (string file in files)
            {
                // Skip files directly in the root folder.
                if (string.Equals(Path.GetDirectoryName(file), folder, StringComparison.OrdinalIgnoreCase))
                {
                    continue;
                }

                string extension = Path.GetExtension(file).ToLower();
                if (extension != ".jpg" && extension != ".png" && extension != ".jpeg")
                {
                    continue;
                }

                string label = useFolderNameAsLabel ? new DirectoryInfo(Path.GetDirectoryName(file)).Name : Path.GetFileName(file);

                yield return new ImageData { ImagePath = file, Label = label };
            }
        }
    }
}
